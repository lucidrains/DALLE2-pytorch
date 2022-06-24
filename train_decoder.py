from pathlib import Path

from dalle2_pytorch.trainer import DecoderTrainer
from dalle2_pytorch.dataloaders import create_image_embedding_dataloader
from dalle2_pytorch.trackers import WandbTracker, ConsoleTracker, DummyTracker
from dalle2_pytorch.train_configs import TrainDecoderConfig
from dalle2_pytorch.utils import Timer, print_ribbon
from dalle2_pytorch.dalle2_pytorch import resize_image_to

import torchvision
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from accelerate import Accelerator, DistributedDataParallelKwargs
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
    embeddings_url,
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
    
    create_dataloader = lambda tar_urls, shuffle=False, resample=False, with_text=False, for_sampling=False: create_image_embedding_dataloader(
        tar_url=tar_urls,
        num_workers=num_workers,
        batch_size=batch_size if not for_sampling else n_sample_images,
        embeddings_url=embeddings_url,
        index_width=index_width,
        shuffle_num = None,
        extra_keys= ["txt"] if with_text else [],
        shuffle_shards = shuffle,
        resample_shards = resample, 
        img_preproc=img_preproc,
        handler=wds.handlers.warn_and_continue
    )

    train_dataloader = create_dataloader(train_urls, shuffle=shuffle_train, resample=resample_train)
    train_sampling_dataloader = create_dataloader(train_urls, shuffle=False, for_sampling=True)
    val_dataloader = create_dataloader(val_urls, shuffle=False, with_text=True)
    test_dataloader = create_dataloader(test_urls, shuffle=False, with_text=True)
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
    embeddings = []
    captions = []
    dataset_keys = get_dataset_keys(dataloader)
    has_caption = "txt" in dataset_keys
    for data in dataloader:
        if has_caption:
            img, emb, txt = data
        else:
            img, emb = data
            txt = [""] * emb.shape[0]
        img = img.to(device=device, dtype=torch.float)
        emb = emb.to(device=device, dtype=torch.float)
        images.extend(list(img))
        embeddings.extend(list(emb))
        captions.extend(list(txt))
        if len(images) >= n:
            break
    return list(zip(images[:n], embeddings[:n], captions[:n]))

def generate_samples(trainer, example_data, text_prepend=""):
    """
    Takes example data and generates images from the embeddings
    Returns three lists: real images, generated images, and captions
    """
    real_images, embeddings, txts = zip(*example_data)
    embeddings_tensor = torch.stack(embeddings)
    samples = trainer.sample(embeddings_tensor)
    generated_images = list(samples)
    captions = [text_prepend + txt for txt in txts]
    return real_images, generated_images, captions

def generate_grid_samples(trainer, examples, text_prepend=""):
    """
    Generates samples and uses torchvision to put them in a side by side grid for easy viewing
    """
    real_images, generated_images, captions = generate_samples(trainer, examples, text_prepend)

    real_image_size = real_images[0].shape[-1]
    generated_image_size = generated_images[0].shape[-1]

    # training images may be larger than the generated one
    if real_image_size > generated_image_size:
        real_images = [resize_image_to(image, generated_image_size) for image in real_images]

    grid_images = [torchvision.utils.make_grid([original_image, generated_image]) for original_image, generated_image in zip(real_images, generated_images)]
    return grid_images, captions
                    
def evaluate_trainer(trainer, dataloader, device, n_evaluation_samples=1000, FID=None, IS=None, KID=None, LPIPS=None):
    """
    Computes evaluation metrics for the decoder
    """
    metrics = {}
    # Prepare the data
    examples = get_example_data(dataloader, device, n_evaluation_samples)
    if len(examples) == 0:
        print("No data to evaluate. Check that your dataloader has shards.")
        return metrics
    real_images, generated_images, captions = generate_samples(trainer, examples)
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
        renorm_real_images = real_images.mul(2).sub(1)
        renorm_generated_images = generated_images.mul(2).sub(1)
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

def save_trainer(tracker, trainer, epoch, sample, next_task, validation_losses, relative_paths):
    """
    Logs the model with an appropriate method depending on the tracker
    """
    if isinstance(relative_paths, str):
        relative_paths = [relative_paths]
    for relative_path in relative_paths:
        local_path = str(tracker.data_path / relative_path)
        trainer.save(local_path, epoch=epoch, sample=sample, next_task=next_task, validation_losses=validation_losses)
        tracker.save_file(local_path)
    
def recall_trainer(tracker, trainer, recall_source=None, **load_config):
    """
    Loads the model with an appropriate method depending on the tracker
    """
    trainer.accelerator.print(print_ribbon(f"Loading model from {recall_source}"))
    local_filepath = tracker.recall_file(recall_source, **load_config)
    state_dict = trainer.load(local_filepath)
    return state_dict.get("epoch", 0), state_dict.get("validation_losses", []), state_dict.get("next_task", "train"), state_dict.get("sample", 0)

def train(
    dataloaders,
    decoder,
    accelerator,
    tracker,
    inference_device,
    load_config=None,
    evaluate_config=None,
    epoch_samples = None,  # If the training dataset is resampling, we have to manually stop an epoch
    validation_samples = None,
    epochs = 20,
    n_sample_images = 5,
    save_every_n_samples = 100000,
    save_all=False,
    save_latest=True,
    save_best=True,
    unet_training_mask=None,
    **kwargs
):
    """
    Trains a decoder on a dataset.
    """
    is_master = accelerator.process_index == 0

    trainer = DecoderTrainer(
        decoder=decoder,
        accelerator=accelerator,
        **kwargs
    )

    # Set up starting model and parameters based on a recalled state dict
    start_epoch = 0
    validation_losses = []
    next_task = 'train'
    sample = 0
    samples_seen = 0
    val_sample = 0
    step = lambda: int(trainer.step.item())

    if exists(load_config) and exists(load_config.source):
        start_epoch, validation_losses, next_task, recalled_sample = recall_trainer(tracker, trainer, recall_source=load_config.source, **load_config.dict())
        if next_task == 'train':
            sample = recalled_sample
        if next_task == 'val':
            val_sample = recalled_sample
        accelerator.print(f"Loaded model from {load_config.source} on epoch {start_epoch} with minimum validation loss {min(validation_losses) if len(validation_losses) > 0 else 'N/A'}")
        accelerator.print(f"Starting training from task {next_task} at sample {sample} and validation sample {val_sample}")
    trainer.to(device=inference_device)

    if not exists(unet_training_mask):
        # Then the unet mask should be true for all unets in the decoder
        unet_training_mask = [True] * trainer.num_unets
    assert len(unet_training_mask) == trainer.num_unets, f"The unet training mask should be the same length as the number of unets in the decoder. Got {len(unet_training_mask)} and {trainer.num_unets}"

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
            for i, (img, emb) in enumerate(dataloaders["train"]):
                # We want to count the total number of samples across all processes
                sample_length_tensor[0] = len(img)
                all_samples = accelerator.gather(sample_length_tensor)  # TODO: accelerator.reduce is broken when this was written. If it is fixed replace this.
                total_samples = all_samples.sum().item()
                sample += total_samples
                samples_seen += total_samples
                img, emb = send_to_device((img, emb))

                trainer.train()
                for unet in range(1, trainer.num_unets+1):
                    # Check if this is a unet we are training
                    if not unet_training_mask[unet-1]: # Unet index is the unet number - 1
                        continue

                    loss = trainer.forward(img, image_embed=emb, unet_number=unet)
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
                    loss_map = { f"Unet {index} Training Loss": loss.item() for index, loss in enumerate(unet_average_loss) if loss != 0 }

                    # gather decay rate on each UNet
                    ema_decay_list = {f"Unet {index} EMA Decay": ema_unet.get_current_decay() for index, ema_unet in enumerate(trainer.ema_unets)}

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
                        tracker.log(log_data, step=step(), verbose=True)

                if is_master and last_snapshot + save_every_n_samples < sample:  # This will miss by some amount every time, but it's not a big deal... I hope
                    # It is difficult to gather this kind of info on the accelerator, so we have to do it on the master
                    print("Saving snapshot")
                    last_snapshot = sample
                    # We need to know where the model should be saved
                    save_paths = []
                    if save_latest:
                        save_paths.append("latest.pth")
                    if save_all:
                        save_paths.append(f"checkpoints/epoch_{epoch}_step_{step()}.pth")
                    save_trainer(tracker, trainer, epoch, sample, next_task, validation_losses, save_paths)
                    if exists(n_sample_images) and n_sample_images > 0:
                        trainer.eval()
                        train_images, train_captions = generate_grid_samples(trainer, train_example_data, "Train: ")
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
            for i, (img, emb, txt) in enumerate(dataloaders["val"]):
                val_sample_length_tensor[0] = len(img)
                all_samples = accelerator.gather(val_sample_length_tensor)
                total_samples = all_samples.sum().item()
                val_sample += total_samples
                img, emb = send_to_device((img, emb))

                for unet in range(1, len(decoder.unets)+1):
                    if not unet_training_mask[unet-1]: # Unet index is the unet number - 1
                        # No need to evaluate an unchanging unet
                        continue
                    
                    loss = trainer.forward(img.float(), image_embed=emb.float(), unet_number=unet)
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
                tracker.log(val_loss_map, step=step(), verbose=True)
            next_task = 'eval'

        if next_task == 'eval':
            if exists(evaluate_config):
                accelerator.print(print_ribbon(f"Starting Evaluation {epoch}", repeat=40))
                evaluation = evaluate_trainer(trainer, dataloaders["val"], inference_device, **evaluate_config.dict())
                if is_master:
                    tracker.log(evaluation, step=step(), verbose=True)
            next_task = 'sample'
            val_sample = 0

        if next_task == 'sample':
            if is_master:
                # Generate examples and save the model if we are the master
                # Generate sample images
                print(print_ribbon(f"Sampling Set {epoch}", repeat=40))
                test_images, test_captions = generate_grid_samples(trainer, test_example_data, "Test: ")
                train_images, train_captions = generate_grid_samples(trainer, train_example_data, "Train: ")
                tracker.log_images(test_images, captions=test_captions, image_section="Test Samples", step=step())
                tracker.log_images(train_images, captions=train_captions, image_section="Train Samples", step=step())

                print(print_ribbon(f"Starting Saving {epoch}", repeat=40))
                # Get the same paths
                save_paths = []
                if save_latest:
                    save_paths.append("latest.pth")
                if all_average_val_losses is not None:
                    average_loss = all_average_val_losses.mean(dim=0).item()
                    if save_best and (len(validation_losses) == 0 or average_loss < min(validation_losses)):
                        save_paths.append("best.pth")
                    validation_losses.append(average_loss)
                save_trainer(tracker, trainer, epoch, sample, next_task, validation_losses, save_paths)
            next_task = 'train'

def create_tracker(accelerator, config, config_path, tracker_type=None, data_path=None):
    """
    Creates a tracker of the specified type and initializes special features based on the full config
    """
    tracker_config = config.tracker
    accelerator_config = {
        "Distributed": accelerator.distributed_type != accelerate_dataclasses.DistributedType.NO,
        "DistributedType": accelerator.distributed_type,
        "NumProcesses": accelerator.num_processes,
        "MixedPrecision": accelerator.mixed_precision
    }
    init_config = { "config": {**config.dict(), **accelerator_config} }
    data_path = data_path or tracker_config.data_path
    tracker_type = tracker_type or tracker_config.tracker_type

    if tracker_type == "dummy":
        tracker = DummyTracker(data_path)
        tracker.init(**init_config)
    elif tracker_type == "console":
        tracker = ConsoleTracker(data_path)
        tracker.init(**init_config)
    elif tracker_type == "wandb":
        # We need to initialize the resume state here
        load_config = config.load
        if load_config.source == "wandb" and load_config.resume:
            # Then we are resuming the run load_config["run_path"]
            run_id = load_config.run_path.split("/")[-1]
            init_config["id"] = run_id
            init_config["resume"] = "must"

        init_config["entity"] = tracker_config.wandb_entity
        init_config["project"] = tracker_config.wandb_project
        tracker = WandbTracker(data_path)
        tracker.init(**init_config)
        tracker.save_file(str(config_path.absolute()), str(config_path.parent.absolute()))
    else:
        raise ValueError(f"Tracker type {tracker_type} not supported by decoder trainer")
    return tracker
    
def initialize_training(config, config_path):
    # Make sure if we are not loading, distributed models are initialized to the same values
    torch.manual_seed(config.seed)

    # Set up accelerator for configurable distributed training
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
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

    # Create the decoder model and print basic info
    decoder = config.decoder.create()
    num_parameters = sum(p.numel() for p in decoder.parameters())

    # Create and initialize the tracker if we are the master
    tracker = create_tracker(accelerator, config, config_path) if rank == 0 else create_tracker(accelerator, config, config_path, tracker_type="dummy")

    accelerator.print(print_ribbon("Loaded Config", repeat=40))
    accelerator.print(f"Running training with {accelerator.num_processes} processes and {accelerator.distributed_type} distributed training")
    accelerator.print(f"Number of parameters: {num_parameters}")
    train(dataloaders, decoder, accelerator,
        tracker=tracker,
        inference_device=accelerator.device,
        load_config=config.load,
        evaluate_config=config.evaluate,
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
