from dalle2_pytorch import Unet, Decoder
from dalle2_pytorch.trainer import DecoderTrainer, print_ribbon
from dalle2_pytorch.dataloaders import create_image_embedding_dataloader
from dalle2_pytorch.trackers import WandbTracker, ConsoleTracker
from dalle2_pytorch.train_configs import TrainDecoderConfig
from dalle2_pytorch.utils import Timer

import json
import torchvision
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
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
    train_split, test_split, val_split = torch.utils.data.random_split(available_shards, [num_train, num_test, num_val], generator=torch.Generator().manual_seed(0))

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


def create_decoder(device, decoder_config, unets_config):
    """Creates a sample decoder"""

    unets = [Unet(**config) for config in unets_config]

    decoder = Decoder(
        unet=unets,
        **decoder_config
    )

    decoder.to(device=device)
    return decoder

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
    print("Generated {} examples".format(len(images)))
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
    grid_images = [torchvision.utils.make_grid([original_image, generated_image]) for original_image, generated_image in zip(real_images, generated_images)]
    return grid_images, captions
                    
def evaluate_trainer(trainer, dataloader, device, n_evalation_samples=1000, FID=None, IS=None, KID=None, LPIPS=None):
    """
    Computes evaluation metrics for the decoder
    """
    metrics = {}
    # Prepare the data
    examples = get_example_data(dataloader, device, n_evalation_samples)
    real_images, generated_images, captions = generate_samples(trainer, examples)
    real_images = torch.stack(real_images).to(device=device, dtype=torch.float)
    generated_images = torch.stack(generated_images).to(device=device, dtype=torch.float)
    # Convert from [0, 1] to [0, 255] and from torch.float to torch.uint8
    int_real_images = real_images.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)
    int_generated_images = generated_images.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)
    if exists(FID):
        fid = FrechetInceptionDistance(**FID)
        fid.to(device=device)
        fid.update(int_real_images, real=True)
        fid.update(int_generated_images, real=False)
        metrics["FID"] = fid.compute().item()
    if exists(IS):
        inception = InceptionScore(**IS)
        inception.to(device=device)
        inception.update(int_real_images)
        is_mean, is_std = inception.compute()
        metrics["IS_mean"] = is_mean.item()
        metrics["IS_std"] = is_std.item()
    if exists(KID):
        kernel_inception = KernelInceptionDistance(**KID)
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
        lpips = LearnedPerceptualImagePatchSimilarity(**LPIPS)
        lpips.to(device=device)
        lpips.update(renorm_real_images, renorm_generated_images)
        metrics["LPIPS"] = lpips.compute().item()
    return metrics

def save_trainer(tracker, trainer, epoch, step, validation_losses, relative_paths):
    """
    Logs the model with an appropriate method depending on the tracker
    """
    if isinstance(relative_paths, str):
        relative_paths = [relative_paths]
    trainer_state_dict = {}
    trainer_state_dict["trainer"] = trainer.state_dict()
    trainer_state_dict['epoch'] = epoch
    trainer_state_dict['step'] = step
    trainer_state_dict['validation_losses'] = validation_losses
    for relative_path in relative_paths:
        tracker.save_state_dict(trainer_state_dict, relative_path)
    
def recall_trainer(tracker, trainer, recall_source=None, **load_config):
    """
    Loads the model with an appropriate method depending on the tracker
    """
    print(print_ribbon(f"Loading model from {recall_source}"))
    state_dict = tracker.recall_state_dict(recall_source, **load_config)
    trainer.load_state_dict(state_dict["trainer"])
    print("Model loaded")
    return state_dict["epoch"], state_dict["step"], state_dict["validation_losses"]

def train(
    dataloaders,
    decoder,
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
    trainer = DecoderTrainer(  # TODO: Change the get_optimizer function so that it can take arbitrary named args so we can just put **kwargs as an argument here
        decoder,
        **kwargs
    )
    # Set up starting model and parameters based on a recalled state dict
    start_step = 0
    start_epoch = 0
    validation_losses = []

    if exists(load_config) and exists(load_config["source"]):
        start_epoch, start_step, validation_losses = recall_trainer(tracker, trainer, recall_source=load_config["source"], **load_config)
    trainer.to(device=inference_device)

    if not exists(unet_training_mask):
        # Then the unet mask should be true for all unets in the decoder
        unet_training_mask = [True] * trainer.num_unets
    assert len(unet_training_mask) == trainer.num_unets, f"The unet training mask should be the same length as the number of unets in the decoder. Got {len(unet_training_mask)} and {trainer.num_unets}"

    print(print_ribbon("Generating Example Data", repeat=40))
    print("This can take a while to load the shard lists...")
    train_example_data = get_example_data(dataloaders["train_sampling"], inference_device, n_sample_images)
    test_example_data = get_example_data(dataloaders["test_sampling"], inference_device, n_sample_images)
    
    send_to_device = lambda arr: [x.to(device=inference_device, dtype=torch.float) for x in arr]
    step = start_step

    for epoch in range(start_epoch, epochs):
        print(print_ribbon(f"Starting epoch {epoch}", repeat=40))

        timer = Timer()

        sample = 0
        last_sample = 0
        last_snapshot = 0

        losses = []

        for i, (img, emb) in enumerate(dataloaders["train"]):
            step += 1
            sample += img.shape[0]
            img, emb = send_to_device((img, emb))
            
            trainer.train()
            for unet in range(1, trainer.num_unets+1):
                # Check if this is a unet we are training
                if not unet_training_mask[unet-1]: # Unet index is the unet number - 1
                    continue

                loss = trainer.forward(img, image_embed=emb, unet_number=unet)
                trainer.update(unet_number=unet)
                losses.append(loss)

            samples_per_sec = (sample - last_sample) / timer.elapsed()

            timer.reset()
            last_sample = sample

            if i % TRAIN_CALC_LOSS_EVERY_ITERS == 0:
                average_loss = sum(losses) / len(losses)
                log_data = {
                    "Training loss": average_loss,
                    "Epoch": epoch,
                    "Sample": sample,
                    "Step": i,
                    "Samples per second": samples_per_sec
                }
                tracker.log(log_data, step=step, verbose=True)
                losses = []

            if last_snapshot + save_every_n_samples < sample:  # This will miss by some amount every time, but it's not a big deal... I hope
                last_snapshot = sample
                # We need to know where the model should be saved
                save_paths = []
                if save_latest:
                    save_paths.append("latest.pth")
                if save_all:
                    save_paths.append(f"checkpoints/epoch_{epoch}_step_{step}.pth")

                save_trainer(tracker, trainer, epoch, step, validation_losses, save_paths)

                if exists(n_sample_images) and n_sample_images > 0:
                    trainer.eval()
                    train_images, train_captions = generate_grid_samples(trainer, train_example_data, "Train: ")
                    tracker.log_images(train_images, captions=train_captions, image_section="Train Samples", step=step)

            if exists(epoch_samples) and sample >= epoch_samples:
                break

        trainer.eval()
        print(print_ribbon(f"Starting Validation {epoch}", repeat=40))
        with torch.no_grad():
            sample = 0
            average_loss = 0
            timer = Timer()
            for i, (img, emb, txt) in enumerate(dataloaders["val"]):
                sample += img.shape[0]
                img, emb = send_to_device((img, emb))
                
                for unet in range(1, len(decoder.unets)+1):
                    loss = trainer.forward(img.float(), image_embed=emb.float(), unet_number=unet)
                    average_loss += loss

                if i % VALID_CALC_LOSS_EVERY_ITERS == 0:
                    print(f"Epoch {epoch}/{epochs} - {sample / timer.elapsed():.2f} samples/sec")
                    print(f"Loss: {average_loss / (i+1)}")
                    print("")

                if exists(validation_samples) and sample >= validation_samples:
                    break

            average_loss /= i+1
            log_data = {
                "Validation loss": average_loss
            }
            tracker.log(log_data, step=step, verbose=True)

        # Compute evaluation metrics
        if exists(evaluate_config):
            print(print_ribbon(f"Starting Evaluation {epoch}", repeat=40))
            evaluation = evaluate_trainer(trainer, dataloaders["val"], inference_device, **evaluate_config)
            tracker.log(evaluation, step=step, verbose=True)

        # Generate sample images
        print(print_ribbon(f"Sampling Set {epoch}", repeat=40))
        test_images, test_captions = generate_grid_samples(trainer, test_example_data, "Test: ")
        train_images, train_captions = generate_grid_samples(trainer, train_example_data, "Train: ")
        tracker.log_images(test_images, captions=test_captions, image_section="Test Samples", step=step)
        tracker.log_images(train_images, captions=train_captions, image_section="Train Samples", step=step)

        print(print_ribbon(f"Starting Saving {epoch}", repeat=40))
        # Get the same paths
        save_paths = []
        if save_latest:
            save_paths.append("latest.pth")
        if save_best and (len(validation_losses) == 0 or average_loss < min(validation_losses)):
            save_paths.append("best.pth")
        validation_losses.append(average_loss)
        save_trainer(tracker, trainer, epoch, step, validation_losses, save_paths)

def create_tracker(config, tracker_type=None, data_path=None, **kwargs):
    """
    Creates a tracker of the specified type and initializes special features based on the full config
    """
    tracker_config = config["tracker"]
    init_config = {}
    init_config["config"] = config.config
    if tracker_type == "console":
        tracker = ConsoleTracker(**init_config)
    elif tracker_type == "wandb":
        # We need to initialize the resume state here
        load_config = config["load"]
        if load_config["source"] == "wandb" and load_config["resume"]:
            # Then we are resuming the run load_config["run_path"]
            run_id = config["resume"]["wandb_run_path"].split("/")[-1]
            init_config["id"] = run_id
            init_config["resume"] = "must"
        init_config["entity"] = tracker_config["wandb_entity"]
        init_config["project"] = tracker_config["wandb_project"]
        tracker = WandbTracker(data_path)
        tracker.init(**init_config)
    else:
        raise ValueError(f"Tracker type {tracker_type} not supported by decoder trainer")
    return tracker
    
def initialize_training(config):
    # Create the save path
    if "cuda" in config["train"]["device"]:
        assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device(config["train"]["device"])
    torch.cuda.set_device(device)
    all_shards = list(range(config["data"]["start_shard"], config["data"]["end_shard"] + 1))

    dataloaders = create_dataloaders (
        available_shards=all_shards,
        img_preproc = config.get_preprocessing(),
        train_prop = config["data"]["splits"]["train"],
        val_prop = config["data"]["splits"]["val"],
        test_prop = config["data"]["splits"]["test"],
        n_sample_images=config["train"]["n_sample_images"],
        **config["data"]
    )

    decoder = create_decoder(device, config["decoder"], config["unets"])
    num_parameters = sum(p.numel() for p in decoder.parameters())
    print(print_ribbon("Loaded Config", repeat=40))
    print(f"Number of parameters: {num_parameters}")

    tracker = create_tracker(config, **config["tracker"])

    train(dataloaders, decoder, 
        tracker=tracker,
        inference_device=device,
        load_config=config["load"],
        evaluate_config=config["evaluate"],
        **config["train"],
    )

# Create a simple click command line interface to load the config and start the training
@click.command()
@click.option("--config_file", default="./train_decoder_config.json", help="Path to config file")
def main(config_file):
    print("Recalling config from {}".format(config_file))
    with open(config_file) as f:
        config = json.load(f)
    config = TrainDecoderConfig(config)
    initialize_training(config)


if __name__ == "__main__":
    main()
