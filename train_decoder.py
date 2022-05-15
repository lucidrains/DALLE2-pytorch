from dalle2_pytorch import Unet, Decoder
from dalle2_pytorch.train import DecoderTrainer
from dalle2_pytorch.dataloaders import create_image_embedding_dataloader
import time
import os
import json
import torchvision
from torchvision import transforms as T
import torch
from torch.cuda.amp import autocast
import webdataset as wds
import wandb
import click


def create_dataloaders(
    available_shards,
    webdataset_base_url,
    embeddings_url,
    shard_width=6,
    num_workers=4,
    batch_size=32,
    shuffle_train=True,
    resample_train=False,
    shuffle_val_test=False,
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
    
    create_dataloader = lambda tar_urls, shuffle=False, resample=False, with_text=False, is_low=False: create_image_embedding_dataloader(
        tar_url=tar_urls,
        num_workers=num_workers,
        batch_size=batch_size if not is_low else min(32, batch_size),
        embeddings_url=embeddings_url,
        index_width=index_width,
        shuffle_num = None,
        extra_keys= ["txt"] if with_text else [],
        shuffle_shards = shuffle,
        resample_shards = False, 
        img_preproc=img_preproc,
        handler=wds.handlers.warn_and_continue
    )

    train_dataloader = create_dataloader(train_urls, shuffle=shuffle_train, resample=resample_train)
    val_dataloader = create_dataloader(val_urls, shuffle=shuffle_val_test, with_text=True)
    test_dataloader = create_dataloader(test_urls, shuffle=shuffle_val_test, with_text=True, is_low=True)
    return {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader
    }


def create_decoder(device, decoder_config, unets_config):
    """Creates a sample decoder"""
    unets = []
    for i in range(0, len(unets_config)):
        unets.append(Unet(
            **unets_config[i]
        ))
        unets[i].to(device=device)
    
    decoder = Decoder(
        unet=tuple(unets),  # Must be tuple because of cast_tuple
        **decoder_config
    )
    decoder.to(device=device)

    return decoder

def generate_samples(decoder, dataloader, epoch, device, step, n=5, text_prepend=""):
    """
    Generates n samples from the decoder and uploads them to wandb
    Consistently uses the first n image embeddings from the dataloader
    If the dataloader returns an extra text key, that is used as the caption
    """
    test_iter = iter(dataloader)
    images = []
    with torch.no_grad():
        decoder.eval()
        for data in test_iter:
            if len(data) == 3:
                img, emb, txt = data
            else:
                img, emb = data
                txt = [""] * emb.shape[0]
            # If the number of images is larger than n, we can remove all embeddings after the first n to speed up inference
            if img.shape[0] > n:
                img = img[:n]
                emb = emb[:n]
                txt = txt[:n]
            img = img.to(device=device, dtype=torch.float)
            emb = emb.to(device=device, dtype=torch.float)
            decoder.to(device=device) # Don't ask me why this is neccesary
            sample = decoder.sample(emb)
            for original_image, generated_image, text in zip(img, sample, txt):
                # Make a grid containing the original image and the generated image
                img_grid = torchvision.utils.make_grid([original_image, generated_image])
                image = wandb.Image(img_grid, caption=text_prepend+text)
                images.append(image)
                
                if len(images) >= n:
                    decoder.to(device)  # Again, don't ask me why but something here moves the model to the cpu
                    decoder.train()
                    return images
    decoder.train()  # In case we run out of samples before n. Your dataset must be tiny, but whatever. Edge cases.

def save_trainer(base_path, trainer, epoch, step, validation_losses, local_only=True, latest=False, best=False, checkpoint=False):
    """
    Saves the state of the trainer and decoder to wandb.
    """
    print("====================================== Saving trainer ======================================")
    state_dict = {}
    state_dict["trainer"] = trainer.state_dict()
    state_dict["decoder"] = trainer.decoder.state_dict()
    state_dict['epoch'] = epoch
    state_dict['step'] = step
    state_dict['validation_losses'] = validation_losses
    save_paths = []
    if latest:
        save_paths.append(os.path.join(base_path, "latest.pth"))
    if best:
        save_paths.append(os.path.join(base_path, "best.pth"))
    if checkpoint:
        save_paths.append(os.path.join(base_path, "checkpoints", f"epoch_{epoch}_step_{step}.pth"))
    for save_path in save_paths:
        print(f"Saving to {save_path}")
        torch.save(state_dict, save_path)
        if not local_only:
            wandb.save(save_path, base_path=base_path)

def recall_trainer(trainer, wandb_run_path=None, wandb_file_path=None, local_filepath=None):
    print(f"====================================== Recall trainer ======================================")
    if local_filepath is None:
        # Then we are recalling from wandb
        print(f"Recalling trainer from wandb: {wandb_run_path}/{wandb_file_path}")
        trainer_state_dict_file = wandb.restore(wandb_file_path, run_path=wandb_run_path)
        state_dict = torch.load(trainer_state_dict_file.name)
    else:
        # Then we are recalling from a file
        print(f"Recalling trainer from file: {local_filepath}")
        state_dict = torch.load(local_filepath)
    trainer.load_state_dict(state_dict["trainer"])
    trainer.decoder.load_state_dict(state_dict["decoder"])
    return trainer, state_dict["epoch"], state_dict["step"], state_dict["validation_losses"] if "validation_losses" in state_dict else []

def train(
    dataloaders,
    decoder,
    inference_device,
    using_wandb=False,
    resume_from=None,
    resume_config=None,
    epoch_samples = None,  # If the training dataset is resampling, we have to manually stop an epoch
    validation_samples = None,
    epochs = 20,
    n_sample_images = 5,
    save_every_n_samples = 100000,
    amp=False,
    lr=1e-4,
    wd=0,
    use_ema=True,
    max_grad_norm=None,
    base_path=None,
    save_all=False,
    save_latest=True,
    save_best=True,
    **kwargs
):
    """
    Trains a decoder on a dataset.
    """
    # Create the base path if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    # If we are saving all, we also need to make a checkpoints folder inside there
    if save_all:
        if not os.path.exists(os.path.join(base_path, "checkpoints")):
            os.makedirs(os.path.join(base_path, "checkpoints"))
    trainer = DecoderTrainer(
        decoder,
        lr = lr,
        wd = wd,
        amp = amp,
        use_ema = use_ema,
        max_grad_norm = max_grad_norm
    )
    # Set up starting model and parameters based on a recalled state dict
    start_step = 0
    start_epoch = 0
    validation_losses = []
    if resume_from is not None:
        if resume_from == "wandb":
            trainer, start_epoch, start_step, validation_losses = recall_trainer(
                trainer,
                wandb_run_path=resume_config["wandb_run_path"],
                wandb_file_path=resume_config["wandb_file_path"],
            )
        elif resume_from == "file":
            trainer, start_epoch, start_step, validation_losses = recall_trainer(
                trainer,
                local_filepath=resume_config["filepath"]
            )
        else:
            raise ValueError("resume_from must be either wandb or file")
    trainer = trainer.to(device=inference_device)
    
    send_to_device = lambda arr: [x.to(device=inference_device, dtype=torch.float) for x in arr]
    step = start_step
    for epoch in range(start_epoch, epochs):
        print(f"=========== Starting epoch {epoch} ===========")
        decoder.train()
        sample = 0
        last_sample = 0
        last_snapshot = 0
        last_time = time.time()
        losses = []
        for i, (img, emb) in enumerate(dataloaders["train"]):
            step += 1
            sample += img.shape[0]
            img, emb = send_to_device((img, emb))
            
            for unet in range(1, trainer.num_unets+1):
                loss = trainer.forward(img, image_embed=emb, unet_number=unet)
                loss.backward()
                trainer.update(unet_number=unet)
                losses.append(loss.item())

            samples_per_sec = (sample - last_sample) / (time.time() - last_time)
            last_time = time.time()
            last_sample = sample

            if i % 10 == 0:
                average_loss = sum(losses) / len(losses)
                print(f"Epoch {epoch}/{epochs} Sample {sample} Step {i} - {samples_per_sec:.2f} samples/sec")
                print(f"Losses: {losses}")
                print(f"Loss: {average_loss}")
                print("")
                if using_wandb:
                    wandb.log({
                        "Training loss": average_loss,
                        "Epoch": epoch,
                        "Sample": sample,
                        "Step": i,
                        "Samples per second": samples_per_sec
                    }, step=step)
                losses = []

            if last_snapshot + save_every_n_samples < sample:
                last_snapshot = sample
                print(f"Saving model...")
                save_trainer(base_path, trainer, epoch, step, validation_losses, local_only=not using_wandb, latest=save_latest, best=False, checkpoint=save_all)
                if using_wandb:
                    print(f"Generating sample...")
                    train_images = generate_samples(decoder, dataloaders["train"], epoch, inference_device, step, n=n_sample_images, text_prepend="Train: ")
                    wandb.log({
                        "Train samples": train_images
                    }, step=step)

            if epoch_samples is not None and sample >= epoch_samples:
                break

        print(f"=========== Starting Validation {epoch} ===========")
        with torch.no_grad():
            decoder.eval()
            sample = 0
            average_loss = 0
            start_time = time.time()
            for i, (img, emb, txt) in enumerate(dataloaders["val"]):
                sample += img.shape[0]
                img, emb = send_to_device((img, emb))
                
                with autocast(enabled=amp):
                    for unet in range(1, len(decoder.unets)+1):
                        loss = decoder.forward(img.float(), image_embed=emb.float(), unet_number=unet)
                        average_loss += loss.item()

                if i % 10 == 0:
                    print(f"Epoch {epoch}/{epochs} - {sample / (time.time() - start_time):.2f} samples/sec")
                    print(f"Loss: {average_loss / (i+1)}")
                    print("")

                if validation_samples is not None and sample >= validation_samples:
                    break
            average_loss /= (i+1)
            print(f"=========== Validation {epoch} Complete ===========")
            print(f"Average Loss: {average_loss}")
            print("")
            if using_wandb:
                wandb.log({
                    "Validation loss": average_loss,
                    "Epoch": epoch,
                    "Sample": sample
                }, step=step)

        if using_wandb:
            print(f"=========== Sampling Set {epoch} ===========")
            test_images = generate_samples(decoder, dataloaders["test"], epoch, inference_device, step, n=n_sample_images, text_prepend="Test: ")
            train_images = generate_samples(decoder, dataloaders["train"], epoch, inference_device, step, n=n_sample_images, text_prepend="Train: ")
            wandb.log({
                "Test samples": test_images,
                "Train samples": train_images
            }, step=step)

        print(f"=========== Starting Saving {epoch} ===========")
        is_best = False
        if save_best:
            is_best = len(validation_losses) == 0 or average_loss < min(validation_losses)
        validation_losses.append(average_loss)
        save_trainer(base_path, trainer, epoch, step, validation_losses, local_only=not using_wandb, latest=True, best=is_best, checkpoint=False)

    
def initialize_training(config):
    using_wandb = len(config["wandb"]["entity"]) > 0 and len(config["wandb"]["project"]) > 0
    resuming = config["resume"]["do_resume"]
    wandb_resume = config["resume"]["wandb_resume"]  # Whether to take over an old wandb run
    resuming_from_source = None if not resuming else ("wandb" if config["resume"]["from_wandb"] else "file")

    # Create the save path
    base_path = "./models"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
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
        **config["data"]
    )

    decoder = create_decoder(device, config["decoder"], config["unets"])

    if using_wandb:
        if wandb_resume:
            run_id = config["resume"]["wandb_run_path"].split("/")[-1]
            wandb.init(
                project=config["wandb"]["project"],
                entity=config["wandb"]["entity"],
                id=run_id,
                resume="must",
                config=config.config,  # :P
            )
        else:
            wandb.init(
                project=config["wandb"]["project"],
                entity=config["wandb"]["entity"],
                config=config.config,
            )

    train(dataloaders, decoder,
        inference_device=device,
        resume_from=resuming_from_source,
        resume_config=config["resume"],
        using_wandb=using_wandb,
        **config["train"],
    )


class TrainDecoderConfig:
    def __init__(self, config):
        defaults = {
            "unets": [{ "dim": 16, "image_emb_dim": 768, "cond_dim": 64, "channels": 3, "dim_mults": [1, 2, 3, 4], "attn_dim_head": 32, "attn_heads": 16 }],
            "decoder": { "image_sizes": [64,], "image_size": [64,], "channels": 3, "timesteps": 1000, "image_cond_drop_prob": 0.1, "text_cond_drop_prob": 0.5, "condition_on_text_encodings": False, "loss_type": "l2", "beta_schedule": "cosine" },
            "data": { "webdataset_base_url": None, "embeddings_url": None, "num_workers": 4, "batch_size": 64, "start_shard": 0, "end_shard": 9999999, "shard_width": None, "index_width": None, "splits": { "train": 0.75, "val": 0.15, "test": 0.1 }, "shuffle_train": True, "resample_train": False, "shuffle_val_test": False, "preprocessing": { "RandomResizedCrop": { "size": [64, 64], "scale": [0.75, 1.0], "ratio": [1.0, 1.0] }, "ToTensor": True } },
            "train": { "epochs": 100, "lr": 2e-5, "wd": 0.0, "max_grad_norm": 0.5, "save_every_n_samples": 100000, "n_sample_images": 16, "device": "cpu", "epoch_samples": None, "validation_samples": None, "use_ema": True, "amp": False, "base_path": None, "save_all": False, "save_latest": True, "save_best": True },
            "wandb": { "entity": "", "project": "" },
            "resume": { "do_resume": False, "from_wandb": True, "wandb_run_path": "", "wandb_file_path": "", "wandb_resume": False, "filepath": "" }
        }
        self.config = self.map_config(config, defaults)

    def map_config(self, config, defaults):
        """
        Returns a dictionary containing all config options in the union of config and defaults.
        If the config value is an array, apply the default value to each element.
        If the default values dict has a value of None for a key, it is required and a runtime error should be thrown if a value is not supplied from config
        """
        def _check_option(option, option_config, option_defaults):
            for key, value in option_defaults.items():
                if key not in option_config:
                    if value is None:
                        raise RuntimeError("Required config value '{}' of option '{}' not supplied".format(key, option))
                    option_config[key] = value
        
        for key, value in defaults.items():
            if key not in config:
                # Then they did not pass in one of the main configs. This will probably result in an error, but we can pass it through since some options have no required values
                if isinstance(value, dict):
                    config[key] = {}
                elif isinstance(value, list):
                    config[key] = [{}]
            # Config[key] is now either a dict or list of dicts. 
            # If it is a list, then we need to check each element
            if isinstance(value, list):
                assert isinstance(config[key], list)
                for element in config[key]:
                    _check_option(key, element, value[0])
            else:
                _check_option(key, config[key], value)
        return config

    def get_preprocessing(self):
        """
        Takes the preprocessing dictionary and converts it to a composition of torchvision transforms
        """
        def _get_transformation(transformation_name, **kwargs):
            if transformation_name == "RandomResizedCrop":
                return T.RandomResizedCrop(**kwargs)
            elif transformation_name == "RandomHorizontalFlip":
                return T.RandomHorizontalFlip()
            elif transformation_name == "ToTensor":
                return T.ToTensor()
        
        transformations = []
        for transformation_name, transformation_kwargs in self.config["data"]["preprocessing"].items():
            if isinstance(transformation_kwargs, dict):
                transformations.append(_get_transformation(transformation_name, **transformation_kwargs))
            else:
                transformations.append(_get_transformation(transformation_name))
        return T.Compose(transformations)
    
    def __getitem__(self, key):
        return self.config[key]

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