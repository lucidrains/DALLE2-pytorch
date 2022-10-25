from pathlib import Path
from typing import List
from datetime import timedelta

from dalle2_pytorch.trainer import DecoderTrainer
from dalle2_pytorch.dataloaders import create_image_embedding_dataloader
from dalle2_pytorch.trackers import Tracker
from dalle2_pytorch.train_configs import DecoderConfig, TrainDecoderConfig
from dalle2_pytorch.utils import Timer, print_ribbon
from dalle2_pytorch.dalle2_pytorch import Decoder, resize_image_to
from clip import tokenize

import torchvision
import torch
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import dataclasses as accelerate_dataclasses
import webdataset as wds
import click

# constants

TRAIN_CALC_LOSS_EVERY_ITERS = 10
VALID_CALC_LOSS_EVERY_ITERS = 10

# helpers functions

def exists(val):
    return val is not None

# main functions

def create_dataloaders(
    available_shards,
    webdataset_base_url,
    img_embeddings_url=None,
    text_embeddings_url=None,
    shard_width=6,
    num_workers=4,
    batch_size=32,
    n_sample_images=6,
    shuffle_train=True,
    resample_train=False,
    img_preproc = None,
    index_width=4,
    train_prop = 0.75,
    val_prop = 0.15,
    test_prop = 0.10,
    seed = 0,
    **kwargs
):
    """
    Randomly splits the available shards into train, val, and test sets and returns a dataloader for each
    """
    assert train_prop + test_prop + val_prop == 1
    num_train = round(train_prop*len(available_shards))
    num_test = round(test_prop*len(available_shards))
    num_val = len(available_shards) - num_train - num_test
    assert num_train + num_test + num_val == len(available_shards), f"{num_train} + {num_test} + {num_val} = {num_train + num_test + num_val} != {len(available_shards)}"
    train_split, test_split, val_split = torch.utils.data.random_split(available_shards, [num_train, num_test, num_val], generator=torch.Generator().manual_seed(seed))

    # The shard number in the webdataset file names has a fixed width. We zero pad the shard numbers so they correspond to a filename.
    train_urls = [webdataset_base_url.format(str(shard).zfill(shard_width)) for shard in train_split]
    test_urls = [webdataset_base_url.format(str(shard).zfill(shard_width)) for shard in test_split]
    val_urls = [webdataset_base_url.format(str(shard).zfill(shard_width)) for shard in val_split]
    
    create_dataloader = lambda tar_urls, shuffle=False, resample=False, for_sampling=False: create_image_embedding_dataloader(
        tar_url=tar_urls,
        num_workers=num_workers,
        batch_size=batch_size if not for_sampling else n_sample_images,
        img_embeddings_url=img_embeddings_url,
        text_embeddings_url=text_embeddings_url,
        index_width=index_width,
        shuffle_num = None,
        extra_keys= ["txt"],
        shuffle_shards = shuffle,
        resample_shards = resample, 
        img_preproc=img_preproc,
        handler=wds.handlers.warn_and_continue
    )

    train_dataloader = create_dataloader(train_urls, shuffle=shuffle_train, resample=resample_train)
    train_sampling_dataloader = create_dataloader(train_urls, shuffle=False, for_sampling=True)
    val_dataloader = create_dataloader(val_urls, shuffle=False)
    test_dataloader = create_dataloader(test_urls, shuffle=False)
    test_sampling_dataloader = create_dataloader(test_urls, shuffle=False, for_sampling=True)
    return {
        "train": train_dataloader,
        "train_sampling": train_sampling_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
        "test_sampling": test_sampling_dataloader
    }

def get_dataset_keys(dataloader):
    """
    It is sometimes neccesary to get the keys the dataloader is returning. Since the dataset is burried in the dataloader, we need to do a process to recover it.
    """
    # If the dataloader is actually a WebLoader, we need to extract the real dataloader
    if isinstance(dataloader, wds.WebLoader):
        dataloader = dataloader.pipeline[0]
    return dataloader.dataset.key_map

def get_example_data(dataloader, device, n=5):
    """
    Samples the dataloader and returns a zipped list of examples
    """
    images = []
    img_embeddings = []
    text_embeddings = []
    captions = []
    for img, emb, txt in dataloader:
        img_emb, text_emb = emb.get('img'), emb.get('text')
        if img_emb is not None:
            img_emb = img_emb.to(device=device, dtype=torch.float)
            img_embeddings.extend(list(img_emb))
        else:
            # Then we add None img.shape[0] times
            img_embeddings.extend([None]*img.shape[0])
        if text_emb is not None:
            text_emb = text_emb.to(device=device, dtype=torch.float)
            text_embeddings.extend(list(text_emb))
        else:
            # Then we add None img.shape[0] times
            text_embeddings.extend([None]*img.shape[0])
        img = img.to(device=device, dtype=torch.float)
        images.extend(list(img))
        captions.extend(list(txt))
        if len(images) >= n:
            break
    return list(zip(images[:n], img_embeddings[:n], text_embeddings[:n], captions[:n]))

def generate_samples(trainer, example_data, clip=None, start_unet=1, end_unet=None, condition_on_text_encodings=False, cond_scale=1.0, device=None, text_prepend="", match_image_size=True):
    """
    Takes example data and generates images from the embeddings
    Returns three lists: real images, generated images, and captions
    """
    real_images, img_embeddings, text_embeddings, txts = zip(*example_data)
    sample_params = {}
    if img_embeddings[0] is None:
        # Generate image embeddings from clip
        imgs_tensor = torch.stack(real_images)
        assert clip is not None, "clip is None, but img_embeddings is None"
        imgs_tensor.to(device=device)
        img_embeddings, img_encoding = clip.embed_image(imgs_tensor)
        sample_params["image_embed"] = img_embeddings
    else:
        # Then we are using precomputed image embeddings
        img_embeddings = torch.stack(img_embeddings)
        sample_params["image_embed"] = img_embeddings
    if condition_on_text_encodings:
        if text_embeddings[0] is None:
            # Generate text embeddings from text
            assert clip is not None, "clip is None, but text_embeddings is None"
            tokenized_texts = tokenize(txts, truncate=True).to(device=device)
            text_embed, text_encodings = clip.embed_text(tokenized_texts)
            sample_params["text_encodings"] = text_encodings
        else:
            # Then we are using precomputed text embeddings
            text_embeddings = torch.stack(text_embeddings)
            sample_params["text_encodings"] = text_embeddings
    sample_params["start_at_unet_number"] = start_unet
    sample_params["stop_at_unet_number"] = end_unet
    if start_unet > 1:
        # If we are only training upsamplers
        sample_params["image"] = torch.stack(real_images)
    if device is not None:
        sample_params["_device"] = device
    samples = trainer.sample(**sample_params, _cast_deepspeed_precision=False)  # At sampling time we don't want to cast to FP16
    generated_images = list(samples)
    captions = [text_prepend + txt for txt in txts]
    if match_image_size:
        generated_image_size = generated_images[0].shape[-1]
        real_images = [resize_image_to(image, generated_image_size, clamp_range=(0, 1)) for image in real_images]
    return real_images, generated_images, captions

def generate_grid_samples(trainer, examples, clip=None, start_unet=1, end_unet=None, condition_on_text_encodings=False, cond_scale=1.0, device=None, text_prepend=""):
    """
    Generates samples and uses torchvision to put them in a side by side grid for easy viewing
    """
    real_images, generated_images, captions = generate_samples(trainer, examples, clip, start_unet, end_unet, condition_on_text_encodings, cond_scale, device, text_prepend)
    grid_images = [torchvision.utils.make_grid([original_image, generated_image]) for original_image, generated_image in zip(real_images, generated_images)]
    return grid_images, captions
                    
def evaluate_trainer(trainer, dataloader, device, start_unet, end_unet, clip=None, condition_on_text_encodings=False, cond_scale=1.0, inference_device=None, n_evaluation_samples=1000, FID=None, IS=None, KID=None, LPIPS=None):
    """
    Computes evaluation metrics for the decoder
    """
    metrics = {}
    # Prepare the data
    examples = get_example_data(dataloader, device, n_evaluation_samples)
    if len(examples) == 0:
        print("No data to evaluate. Check that your dataloader has shards.")
        return metrics
    real_images, generated_images, captions = generate_samples(trainer, examples, clip, start_unet, end_unet, condition_on_text_encodings, cond_scale, inference_device)
    real_images = torch.stack(real_images).to(device=device, dtype=torch.float)
    generated_images = torch.stack(generated_images).to(device=device, dtype=torch.float)
    # Convert from [0, 1] to [0, 255] and from torch.float to torch.uint8
    int_real_images = real_images.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)
    int_generated_images = generated_images.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)

    def null_sync(t, *args, **kwargs):
        return [t]

    if exists(FID):
        fid = FrechetInceptionDistance(**FID, dist_sync_fn=null_sync)
        fid.to(device=device)
        fid.update(int_real_images, real=True)
        fid.update(int_generated_images, real=False)
        metrics["FID"] = fid.compute().item()
    if exists(IS):
        inception = InceptionScore(**IS, dist_sync_fn=null_sync)
        inception.to(device=device)
        inception.update(int_real_images)
        is_mean, is_std = inception.compute()
        metrics["IS_mean"] = is_mean.item()
        metrics["IS_std"] = is_std.item()
    if exists(KID):
        kernel_inception = KernelInceptionDistance(**KID, dist_sync_fn=null_sync)
        kernel_inception.to(device=device)
        kernel_inception.update(int_real_images, real=True)
        kernel_inception.update(int_generated_images, real=False)
        kid_mean, kid_std = kernel_inception.compute()
        metrics["KID_mean"] = kid_mean.item()
        metrics["KID_std"] = kid_std.item()
    if exists(LPIPS):
        # Convert from [0, 1] to [-1, 1]
        renorm_real_images = real_images.mul(2).sub(1).clamp(-1,1)
        renorm_generated_images = generated_images.mul(2).sub(1).clamp(-1,1)
        lpips = LearnedPerceptualImagePatchSimilarity(**LPIPS, dist_sync_fn=null_sync)
        lpips.to(device=device)
        lpips.update(renorm_real_images, renorm_generated_images)
        metrics["LPIPS"] = lpips.compute().item()

    if trainer.accelerator.num_processes > 1:
        # Then we should sync the metrics
        metrics_order = sorted(metrics.keys())
        metrics_tensor = torch.zeros(1, len(metrics), device=device, dtype=torch.float)
        for i, metric_name in enumerate(metrics_order):
            metrics_tensor[0, i] = metrics[metric_name]
        metrics_tensor = trainer.accelerator.gather(metrics_tensor)
        metrics_tensor = metrics_tensor.mean(dim=0)
        for i, metric_name in enumerate(metrics_order):
            metrics[metric_name] = metrics_tensor[i].item()
    return metrics

def save_trainer(tracker: Tracker, trainer: DecoderTrainer, epoch: int, sample: int, next_task: str, validation_losses: List[float], samples_seen: int, is_latest=True, is_best=False):
    """
    Logs the model with an appropriate method depending on the tracker
    """
    tracker.save(trainer, is_best=is_best, is_latest=is_latest, epoch=epoch, sample=sample, next_task=next_task, validation_losses=validation_losses, samples_seen=samples_seen)
    
def recall_trainer(tracker: Tracker, trainer: DecoderTrainer):
    """
    Loads the model with an appropriate method depending on the tracker
    """
    trainer.accelerator.print(print_ribbon(f"Loading model from {type(tracker.loader).__name__}"))
    state_dict = tracker.recall()
    trainer.load_state_dict(state_dict, only_model=False, strict=True)
    return state_dict.get("epoch", 0), state_dict.get("validation_losses", []), state_dict.get("next_task", "train"), state_dict.get("sample", 0), state_dict.get("samples_seen", 0)

def train(
    dataloaders,
    decoder: Decoder,
    accelerator: Accelerator,
    tracker: Tracker,
    inference_device,
    clip=None,
    evaluate_config=None,
    epoch_samples = None,  # If the training dataset is resampling, we have to manually stop an epoch
    validation_samples = None,
    save_immediately=False,
    epochs = 20,
    n_sample_images = 5,
    save_every_n_samples = 100000,
    unet_training_mask=None,
    condition_on_text_encodings=False,
    cond_scale=1.0,
    **kwargs
):
    """
    Trains a decoder on a dataset.
    """
    is_master = accelerator.process_index == 0

    if not exists(unet_training_mask):
        # Then the unet mask should be true for all unets in the decoder
        unet_training_mask = [True] * len(decoder.unets)
    assert len(unet_training_mask) == len(decoder.unets), f"The unet training mask should be the same length as the number of unets in the decoder. Got {len(unet_training_mask)} and {trainer.num_unets}"
    trainable_unet_numbers = [i+1 for i, trainable in enumerate(unet_training_mask) if trainable]
    first_trainable_unet = trainable_unet_numbers[0]
    last_trainable_unet = trainable_unet_numbers[-1]
    def move_unets(unet_training_mask):
        for i in range(len(decoder.unets)):
            if not unet_training_mask[i]:
                # Replace the unet from the module list with a nn.Identity(). This training script never uses unets that aren't being trained so this is fine.
                decoder.unets[i] = nn.Identity().to(inference_device)
    # Remove non-trainable unets
    move_unets(unet_training_mask)

    trainer = DecoderTrainer(
        decoder=decoder,
        accelerator=accelerator,
        dataloaders=dataloaders,
        **kwargs
    )

    # Set up starting model and parameters based on a recalled state dict
    start_epoch = 0
    validation_losses = []
    next_task = 'train'
    sample = 0
    samples_seen = 0
    val_sample = 0
    step = lambda: int(trainer.num_steps_taken(unet_number=first_trainable_unet))

    if tracker.can_recall:
        start_epoch, validation_losses, next_task, recalled_sample, samples_seen = recall_trainer(tracker, trainer)
        if next_task == 'train':
            sample = recalled_sample
        if next_task == 'val':
            val_sample = recalled_sample
        accelerator.print(f"Loaded model from {type(tracker.loader).__name__} on epoch {start_epoch} having seen {samples_seen} samples with minimum validation loss {min(validation_losses) if len(validation_losses) > 0 else 'N/A'}")
        accelerator.print(f"Starting training from task {next_task} at sample {sample} and validation sample {val_sample}")
    trainer.to(device=inference_device)

    accelerator.print(print_ribbon("Generating Example Data", repeat=40))
    accelerator.print("This can take a while to load the shard lists...")
    if is_master:
        train_example_data = get_example_data(dataloaders["train_sampling"], inference_device, n_sample_images)
        accelerator.print("Generated training examples")
        test_example_data = get_example_data(dataloaders["test_sampling"], inference_device, n_sample_images)
        accelerator.print("Generated testing examples")
    
    send_to_device = lambda arr: [x.to(device=inference_device, dtype=torch.float) for x in arr]

    sample_length_tensor = torch.zeros(1, dtype=torch.int, device=inference_device)
    unet_losses_tensor = torch.zeros(TRAIN_CALC_LOSS_EVERY_ITERS, trainer.num_unets, dtype=torch.float, device=inference_device)
    for epoch in range(start_epoch, epochs):
        accelerator.print(print_ribbon(f"Starting epoch {epoch}", repeat=40))

        timer = Timer()
        last_sample = sample
        last_snapshot = sample

        if next_task == 'train':
            for i, (img, emb, txt) in enumerate(dataloaders["train"]):
                # We want to count the total number of samples across all processes
                sample_length_tensor[0] = len(img)
                all_samples = accelerator.gather(sample_length_tensor)  # TODO: accelerator.reduce is broken when this was written. If it is fixed replace this.
                total_samples = all_samples.sum().item()
                sample += total_samples
                samples_seen += total_samples
                img_emb = emb.get('img')
                has_img_embedding = img_emb is not None
                if has_img_embedding:
                    img_emb, = send_to_device((img_emb,))
                text_emb = emb.get('text')
                has_text_embedding = text_emb is not None
                if has_text_embedding:
                    text_emb, = send_to_device((text_emb,))
                img, = send_to_device((img,))

                trainer.train()
                for unet in range(1, trainer.num_unets+1):
                    # Check if this is a unet we are training
                    if not unet_training_mask[unet-1]: # Unet index is the unet number - 1
                        continue

                    forward_params = {}
                    if has_img_embedding:
                        forward_params['image_embed'] = img_emb
                    else:
                        # Forward pass automatically generates embedding
                        assert clip is not None
                        img_embed, img_encoding = clip.embed_image(img)
                        forward_params['image_embed'] = img_embed
                    if condition_on_text_encodings:
                        if has_text_embedding:
                            forward_params['text_encodings'] = text_emb
                        else:
                            # Then we need to pass the text instead
                            assert clip is not None
                            tokenized_texts = tokenize(txt, truncate=True).to(inference_device)
                            assert tokenized_texts.shape[0] == len(img), f"The number of texts ({tokenized_texts.shape[0]}) should be the same as the number of images ({len(img)})"
                            text_embed, text_encodings = clip.embed_text(tokenized_texts)
                            forward_params['text_encodings'] = text_encodings
                    loss = trainer.forward(img, **forward_params, unet_number=unet, _device=inference_device)
                    trainer.update(unet_number=unet)
                    unet_losses_tensor[i % TRAIN_CALC_LOSS_EVERY_ITERS, unet-1] = loss
                
                samples_per_sec = (sample - last_sample) / timer.elapsed()
                timer.reset()
                last_sample = sample

                if i % TRAIN_CALC_LOSS_EVERY_ITERS == 0:
                    # We want to average losses across all processes
                    unet_all_losses = accelerator.gather(unet_losses_tensor)
                    mask = unet_all_losses != 0
                    unet_average_loss = (unet_all_losses * mask).sum(dim=0) / mask.sum(dim=0)
                    loss_map = { f"Unet {index} Training Loss": loss.item() for index, loss in enumerate(unet_average_loss) if unet_training_mask[index] }

                    # gather decay rate on each UNet
                    ema_decay_list = {f"Unet {index} EMA Decay": ema_unet.get_current_decay() for index, ema_unet in enumerate(trainer.ema_unets) if unet_training_mask[index]}

                    log_data = {
                        "Epoch": epoch,
                        "Sample": sample,
                        "Step": i,
                        "Samples per second": samples_per_sec,
                        "Samples Seen": samples_seen,
                        **ema_decay_list,
                        **loss_map
                    }

                    if is_master:
                        tracker.log(log_data, step=step())

                if is_master and (last_snapshot + save_every_n_samples < sample or (save_immediately and i == 0)):  # This will miss by some amount every time, but it's not a big deal... I hope
                    # It is difficult to gather this kind of info on the accelerator, so we have to do it on the master
                    print("Saving snapshot")
                    last_snapshot = sample
                    # We need to know where the model should be saved
                    save_trainer(tracker, trainer, epoch, sample, next_task, validation_losses, samples_seen)
                    if exists(n_sample_images) and n_sample_images > 0:
                        trainer.eval()
                        train_images, train_captions = generate_grid_samples(trainer, train_example_data, clip, first_trainable_unet, last_trainable_unet, condition_on_text_encodings, cond_scale, inference_device, "Train: ")
                        tracker.log_images(train_images, captions=train_captions, image_section="Train Samples", step=step())
                
                if epoch_samples is not None and sample >= epoch_samples:
                    break
            next_task = 'val'
            sample = 0

        all_average_val_losses = None
        if next_task == 'val':
            trainer.eval()
            accelerator.print(print_ribbon(f"Starting Validation {epoch}", repeat=40))
            last_val_sample = val_sample
            val_sample_length_tensor = torch.zeros(1, dtype=torch.int, device=inference_device)
            average_val_loss_tensor = torch.zeros(1, trainer.num_unets, dtype=torch.float, device=inference_device)
            timer = Timer()
            accelerator.wait_for_everyone()
            i = 0
            for i, (img, emb, txt) in enumerate(dataloaders['val']):  # Use the accelerate prepared loader
                val_sample_length_tensor[0] = len(img)
                all_samples = accelerator.gather(val_sample_length_tensor)
                total_samples = all_samples.sum().item()
                val_sample += total_samples
                img_emb = emb.get('img')
                has_img_embedding = img_emb is not None
                if has_img_embedding:
                    img_emb, = send_to_device((img_emb,))
                text_emb = emb.get('text')
                has_text_embedding = text_emb is not None
                if has_text_embedding:
                    text_emb, = send_to_device((text_emb,))
                img, = send_to_device((img,))

                for unet in range(1, len(decoder.unets)+1):
                    if not unet_training_mask[unet-1]: # Unet index is the unet number - 1
                        # No need to evaluate an unchanging unet
                        continue
                        
                    forward_params = {}
                    if has_img_embedding:
                        forward_params['image_embed'] = img_emb.float()
                    else:
                        # Forward pass automatically generates embedding
                        assert clip is not None
                        img_embed, img_encoding = clip.embed_image(img)
                        forward_params['image_embed'] = img_embed
                    if condition_on_text_encodings:
                        if has_text_embedding:
                            forward_params['text_encodings'] = text_emb.float()
                        else:
                            # Then we need to pass the text instead
                            assert clip is not None
                            tokenized_texts = tokenize(txt, truncate=True).to(device=inference_device)
                            assert tokenized_texts.shape[0] == len(img), f"The number of texts ({tokenized_texts.shape[0]}) should be the same as the number of images ({len(img)})"
                            text_embed, text_encodings = clip.embed_text(tokenized_texts)
                            forward_params['text_encodings'] = text_encodings
                    loss = trainer.forward(img.float(), **forward_params, unet_number=unet, _device=inference_device)
                    average_val_loss_tensor[0, unet-1] += loss

                if i % VALID_CALC_LOSS_EVERY_ITERS == 0:
                    samples_per_sec = (val_sample - last_val_sample) / timer.elapsed()
                    timer.reset()
                    last_val_sample = val_sample
                    accelerator.print(f"Epoch {epoch}/{epochs} Val Step {i} -  Sample {val_sample} - {samples_per_sec:.2f} samples/sec")
                    accelerator.print(f"Loss: {(average_val_loss_tensor / (i+1))}")
                    accelerator.print("")
                
                if validation_samples is not None and val_sample >= validation_samples:
                    break
            print(f"Rank {accelerator.state.process_index} finished validation after {i} steps")
            accelerator.wait_for_everyone()
            average_val_loss_tensor /= i+1
            # Gather all the average loss tensors
            all_average_val_losses = accelerator.gather(average_val_loss_tensor)
            if is_master:
                unet_average_val_loss = all_average_val_losses.mean(dim=0)
                val_loss_map = { f"Unet {index} Validation Loss": loss.item() for index, loss in enumerate(unet_average_val_loss) if loss != 0 }
                tracker.log(val_loss_map, step=step())
            next_task = 'eval'

        if next_task == 'eval':
            if exists(evaluate_config):
                accelerator.print(print_ribbon(f"Starting Evaluation {epoch}", repeat=40))
                evaluation = evaluate_trainer(trainer, dataloaders["val"], inference_device, first_trainable_unet, last_trainable_unet, clip=clip, inference_device=inference_device, **evaluate_config.dict(), condition_on_text_encodings=condition_on_text_encodings, cond_scale=cond_scale)
                if is_master:
                    tracker.log(evaluation, step=step())
            next_task = 'sample'
            val_sample = 0

        if next_task == 'sample':
            if is_master:
                # Generate examples and save the model if we are the master
                # Generate sample images
                print(print_ribbon(f"Sampling Set {epoch}", repeat=40))
                test_images, test_captions = generate_grid_samples(trainer, test_example_data, clip, first_trainable_unet, last_trainable_unet, condition_on_text_encodings, cond_scale, inference_device, "Test: ")
                train_images, train_captions = generate_grid_samples(trainer, train_example_data, clip, first_trainable_unet, last_trainable_unet, condition_on_text_encodings, cond_scale, inference_device, "Train: ")
                tracker.log_images(test_images, captions=test_captions, image_section="Test Samples", step=step())
                tracker.log_images(train_images, captions=train_captions, image_section="Train Samples", step=step())

                print(print_ribbon(f"Starting Saving {epoch}", repeat=40))
                is_best = False
                if all_average_val_losses is not None:
                    average_loss = all_average_val_losses.mean(dim=0).sum() / sum(unet_training_mask)
                    if len(validation_losses) == 0 or average_loss < min(validation_losses):
                        is_best = True
                    validation_losses.append(average_loss)
                save_trainer(tracker, trainer, epoch, sample, next_task, validation_losses, samples_seen, is_best=is_best)
            next_task = 'train'

def create_tracker(accelerator: Accelerator, config: TrainDecoderConfig, config_path: str, dummy: bool = False) -> Tracker:
    tracker_config = config.tracker
    accelerator_config = {
        "Distributed": accelerator.distributed_type != accelerate_dataclasses.DistributedType.NO,
        "DistributedType": accelerator.distributed_type,
        "NumProcesses": accelerator.num_processes,
        "MixedPrecision": accelerator.mixed_precision
    }
    accelerator.wait_for_everyone()  # If nodes arrive at this point at different times they might try to autoresume the current run which makes no sense and will cause errors
    tracker: Tracker = tracker_config.create(config, accelerator_config, dummy_mode=dummy)
    tracker.save_config(config_path, config_name='decoder_config.json')
    tracker.add_save_metadata(state_dict_key='config', metadata=config.dict())
    return tracker
    
def initialize_training(config: TrainDecoderConfig, config_path):
    # Make sure if we are not loading, distributed models are initialized to the same values
    torch.manual_seed(config.seed)

    # Set up accelerator for configurable distributed training
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=config.train.find_unused_parameters, static_graph=config.train.static_graph)
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=60*60))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, init_kwargs])

    if accelerator.num_processes > 1:
        # We are using distributed training and want to immediately ensure all can connect
        accelerator.print("Waiting for all processes to connect...")
        accelerator.wait_for_everyone()
        accelerator.print("All processes online and connected")

    # If we are in deepspeed fp16 mode, we must ensure learned variance is off
    if accelerator.mixed_precision == "fp16" and accelerator.distributed_type == accelerate_dataclasses.DistributedType.DEEPSPEED and config.decoder.learned_variance:
        raise ValueError("DeepSpeed fp16 mode does not support learned variance")
    
    # Set up data
    all_shards = list(range(config.data.start_shard, config.data.end_shard + 1))
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    shards_per_process = len(all_shards) // world_size
    assert shards_per_process > 0, "Not enough shards to split evenly"
    my_shards = all_shards[rank * shards_per_process: (rank + 1) * shards_per_process]
    dataloaders = create_dataloaders (
        available_shards=my_shards,
        img_preproc = config.data.img_preproc,
        train_prop = config.data.splits.train,
        val_prop = config.data.splits.val,
        test_prop = config.data.splits.test,
        n_sample_images=config.train.n_sample_images,
        **config.data.dict(),
        rank = rank,
        seed = config.seed,
    )

    # If clip is in the model, we need to remove it for compatibility with deepspeed
    clip = None
    if config.decoder.clip is not None:
        clip = config.decoder.clip.create()  # Of course we keep it to use it during training, just not in the decoder as that causes issues
        config.decoder.clip = None
    # Create the decoder model and print basic info
    decoder = config.decoder.create()
    get_num_parameters = lambda model, only_training=False: sum(p.numel() for p in model.parameters() if (p.requires_grad or not only_training))

    # Create and initialize the tracker if we are the master
    tracker = create_tracker(accelerator, config, config_path, dummy = rank!=0)

    has_img_embeddings = config.data.img_embeddings_url is not None
    has_text_embeddings = config.data.text_embeddings_url is not None
    conditioning_on_text = any([unet.cond_on_text_encodings for unet in config.decoder.unets])

    has_clip_model = clip is not None
    data_source_string = ""

    if has_img_embeddings:
        data_source_string += "precomputed image embeddings"
    elif has_clip_model:
        data_source_string += "clip image embeddings generation"
    else:
        raise ValueError("No image embeddings source specified")
    if conditioning_on_text:
        if has_text_embeddings:
            data_source_string += " and precomputed text embeddings"
        elif has_clip_model:
            data_source_string += " and clip text encoding generation"
        else:
            raise ValueError("No text embeddings source specified")

    accelerator.print(print_ribbon("Loaded Config", repeat=40))
    accelerator.print(f"Running training with {accelerator.num_processes} processes and {accelerator.distributed_type} distributed training")
    accelerator.print(f"Training using {data_source_string}. {'conditioned on text' if conditioning_on_text else 'not conditioned on text'}")
    accelerator.print(f"Number of parameters: {get_num_parameters(decoder)} total; {get_num_parameters(decoder, only_training=True)} training")
    for i, unet in enumerate(decoder.unets):
        accelerator.print(f"Unet {i} has {get_num_parameters(unet)} total; {get_num_parameters(unet, only_training=True)} training")

    train(dataloaders, decoder, accelerator,
        clip=clip,
        tracker=tracker,
        inference_device=accelerator.device,
        evaluate_config=config.evaluate,
        condition_on_text_encodings=conditioning_on_text,
        **config.train.dict(),
    )
    
# Create a simple click command line interface to load the config and start the training
@click.command()
@click.option("--config_file", default="./train_decoder_config.json", help="Path to config file")
def main(config_file):
    config_file_path = Path(config_file)
    config = TrainDecoderConfig.from_json_path(str(config_file_path))
    initialize_training(config, config_path=config_file_path)

if __name__ == "__main__":
    main()
