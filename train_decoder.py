from dalle2_pytorch import Unet, Decoder
from dalle2_pytorch.train import DecoderTrainer
from dalle2_pytorch.optimizer import get_optimizer
from dalle2_pytorch.dataloaders import create_image_embedding_dataloader, ImageEmbeddingDataset
import time
import torchvision
from torchvision import transforms as T
import torch
from torch.cuda.amp import autocast,GradScaler
import webdataset as wds
import fsspec
import wandb
import os

def create_dataloaders(
    available_shards,
    webdataset_base_url,
    embeddings_url,
    num_workers=4,
    batch_size=32,
    shuffle_train=True,
    shuffle_val_test=False,
    img_size=(256, 256),
    img_preproc = None,
    index_width=4,
    train_prop = 0.75,
    val_prop = 0.15,
    test_prop = 0.10
):
    assert train_prop + test_prop + val_prop == 1
    num_train = round(train_prop*len(available_shards))
    num_test = round(test_prop*len(available_shards))
    num_val = round(val_prop*len(available_shards))
    assert num_train + num_test + num_val == len(available_shards)
    train_split, test_split, val_split = torch.utils.data.random_split(available_shards, [num_train, num_test, num_val], generator=torch.Generator().manual_seed(0))

    train_urls = [webdataset_base_url.format(str(shard).zfill(6)) for shard in train_split]
    test_urls = [webdataset_base_url.format(str(shard).zfill(6)) for shard in test_split]
    val_urls = [webdataset_base_url.format(str(shard).zfill(6)) for shard in val_split]

    if img_preproc is None:
        img_preproc = T.Compose([
            T.RandomResizedCrop(img_size,
                                scale=(0.75, 1.),
                                ratio=(1., 1.)),
            T.ToTensor(),
        ])
    
    create_dataloader = lambda tar_urls, shuffle, with_text=False, is_single=False: create_image_embedding_dataloader(
        tar_url=tar_urls,
        num_workers=num_workers,
        batch_size=batch_size if not is_single else 1,
        embeddings_url=embeddings_url,
        index_width=index_width,
        shuffle_num = None,
        extra_keys= ["txt"] if with_text else [],
        shuffle_shards = shuffle,
        resample_shards = False, 
        img_preproc=img_preproc,
        handler=wds.handlers.warn_and_continue
    )

    train_dataloader = create_dataloader(train_urls, shuffle_train)
    val_dataloader = create_dataloader(val_urls, shuffle_val_test, with_text=True)
    test_dataloader = create_dataloader(test_urls, shuffle_val_test, with_text=True, is_single=True)
    return {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader
    }


def create_decoder(device):
    """Creates a sample decoder"""
    unet_conifg = [{
        "dim": 16,
        "image_embed_dim": 768,
        "cond_dim": 64,
        "channels": 3,
        "dim_mults": (1, 2, 3, 4),
        "attn_dim_head": 64,
        "attn_heads": 32,
        "lowres_cond": True,
        "loss_type": "l1"  # l1, l2, huber
    }]
    unet1 = Unet(
        **unet_conifg[0],
    )
    unet1.to(device)

    decoder_config = {
        "unet": (unet1,),
        "image_sizes": (64,),
        "image_size": (64,),
        "channels": 3,
        "timesteps": 1000,
        "image_cond_drop_prob": 0.1,
        "text_cond_drop_prob": 0.5,
        "condition_on_text_encodings": False  # set this to True if you wish to condition on text during training and sampling
    }
    decoder = Decoder(
        **decoder_config
    )
    decoder.to(device)

    return decoder, (unet_conifg, decoder_config)

def generate_samples(decoder, dataloader, epoch, device, step, n=5, text_prepend=""):
    test_iter = iter(dataloader)
    images = []
    with torch.no_grad():
        for i in range(n):
            data = next(test_iter)
            if len(data) == 3:
                img, emb, txt = data
            else:
                img, emb = data
                txt = [""] * emb.shape[0]
            img = img.to(device=device, dtype=torch.float)
            emb = emb.to(device=device, dtype=torch.float)
            sample = decoder.sample(emb)
            for original_image, generated_image, text in zip(img, sample, txt):
                # Make a grid containing the original image and the generated image
                img_grid = torchvision.utils.make_grid([original_image, generated_image])
                image = wandb.Image(img_grid, caption=text_prepend+text)
                images.append(image)
                break
    decoder.to(device)
    return images

def save_trainer(save_folder, trainer, epoch, step):
    # Saves the trainer state_dict
    print("====================================== Saving trainer ======================================")
    state_dict = trainer.state_dict()
    state_dict['epoch'] = epoch
    state_dict['step'] = step
    filename = f"trainer_epoch_{epoch}_step_{step}.pth"
    file_path = os.path.join(save_folder, filename)
    torch.save(state_dict, file_path)
    # Save to wandb
    wandb.save(file_path)

def overfit(
    dataloader,
    decoder,
    epoch_length,
    epochs,
    n_samples,
    device,
    **kwargs
):
    print("Overfitting")
    trainer = DecoderTrainer(
        decoder,
        lr=4e-4,
        **kwargs
    )
    send_to_device = lambda arr: [x.to(device=device, dtype=torch.float) for x in arr]
    step = 0
    for epoch in range(epochs):
        start_time = time.time()
        sample = 0
        decoder.train()
        data = next(iter(dataloader))
        if len(data) == 3:
            img, emb, txt = data
        else:
            img, emb = data
            txt = None
        img, emb = send_to_device((img, emb))
        for i in range(epoch_length):
            sample += img.shape[0]
            step += 1
            losses = []
            for unet in range(1, trainer.num_unets+1):
                loss = trainer.forward(img, image_embed=emb, unet_number=unet)
                loss.backward()
                trainer.update(unet_number=unet)
                losses.append(loss.item())
            samples_per_sec = sample / (time.time() - start_time)

            if (i + 1) % 10 == 0:
                print(f"Overfit Epoch {epoch}/{epochs} Sample {sample} Step {i} - {samples_per_sec:.2f} samples/sec")
                print(f"Losses: {losses}")
                print(f"Loss: {sum(losses)}")
                print("")
            wandb.log({
                "Training loss": sum(losses) / len(losses),
                "Epoch": epoch,
                "Sample": sample,
                "Step": i,
                "Samples per second": samples_per_sec
            }, step=step)

            if (i + 1) % min(800, epoch_length) == 0:
                print(f"Generating sample...")
                images = generate_samples(decoder, dataloader, epoch, device, step, n=n_samples)
                wandb.log({"Samples": images}, step=step)


def train(
    dataloaders,
    decoder,
    save_path,
    device,
    epoch_length = None,  # If the training dataset is resampling, we have to manually stop an epoch
    validation_length = None,
    epochs = 20,
    n_samples = 5,
    amp=False,
    learning_rate = 4e-4,
    weight_decay = 0,
    **kwargs
):
    trainer = DecoderTrainer(
        decoder,
        lr=learning_rate,
        wd=weight_decay,
        **kwargs
    )

    # scaler = GradScaler(enabled=amp)
    # optimizer = get_optimizer(decoder.parameters(), wd=1e-2, lr=1.2e-4)

    # if dataloaders["train"].dataset.resampling:
    #     assert epoch_length is not None, "If training with resampling, a maximum epoch length must be set"
    # Send an array of tensors to the device
    send_to_device = lambda arr: [x.to(device=device, dtype=torch.float) for x in arr]
    step = 0
    for epoch in range(epochs):
        print(f"=========== Starting epoch {epoch} ===========")
        decoder.train()
        sample = 0
        start_time = time.time()
        for i, (img, emb) in enumerate(dataloaders["train"]):
            step += 1
            sample += img.shape[0]
            img, emb = send_to_device((img, emb))
            
            losses = []
            for unet in range(1, trainer.num_unets+1):
                loss = trainer.forward(img.float(), image_embed=emb.float(), unet_number=unet)
                loss.backward()
                trainer.update(unet_number=unet)
                losses.append(loss.item())

            samples_per_sec = sample / (time.time() - start_time)

            if i % 10 == 0:
                print(f"Epoch {epoch}/{epochs} Sample {sample} Step {i} - {samples_per_sec:.2f} samples/sec")
                print(f"Losses: {losses}")
                print(f"Loss: {sum(losses)}")
                print("")
            wandb.log({
                "Training loss": sum(losses) / len(losses),
                "Epoch": epoch,
                "Sample": sample,
                "Step": i,
                "Samples per second": samples_per_sec
            }, step=step)

            if i % 4000 == 0:
                print(f"Saving model...")
                save_trainer(save_path, trainer, epoch, step)

            if i % 800 == 0:
                print(f"Generating sample...")
                test_images = generate_samples(decoder, dataloaders["test"], epoch, device, step, n=2, text_prepend="Test: ")
                train_images = generate_samples(decoder, dataloaders["train"], epoch, device, step, n=2, text_prepend="Train: ")
                wandb.log({
                    "Test samples": test_images,
                    "Train samples": train_images
                }, step=step)

            if epoch_length is not None and i >= epoch_length:
                break

        print(f"=========== Starting Validation {epoch} ===========")
        with torch.no_grad():
            decoder.eval()
            sample = 0
            average_loss = 0
            for i, (img, emb, txt) in enumerate(dataloaders["val"]):
                sample += img.shape[0]
                img, emb = send_to_device((img, emb))
                
                # for unet in range(1, trainer.num_unets+1):
                #     loss = trainer.forward(img.float(), image_embed=emb.float(), unet_number=unet)
                #     average_loss += loss.item()
                with autocast(enabled=amp):
                    for unet in range(1, len(decoder.unets)+1):
                        loss = decoder.forward(img.float(), image_embed=emb.float(), unet_number=unet)
                        average_loss += loss.item()

                if i % 10 == 0:
                    print(f"Epoch {epoch}/{epochs} - {sample / (time.time() - start_time):.2f} samples/sec")
                    print(f"Loss: {average_loss / (i+1)}")
                    print("")

                if validation_length is not None and i >= validation_length:
                    break
            average_loss /= sample
            wandb.log({
                "Validation loss": average_loss,
                "Epoch": epoch,
                "Sample": sample
            }, step=step)

        print(f"=========== Starting Sampling {epoch} ===========")
        if n_samples is not None and n_samples > 0:
            # Then we will select the top N samples from test and use trainer.sample(image_embed) to generate samples
            test_images = generate_samples(decoder, dataloaders["test"], epoch, device, step, n=n_samples, text_prepend="Test: ")
            train_images = generate_samples(decoder, dataloaders["train"], epoch, device, step, n=n_samples, text_prepend="Train: ")
            wandb.log({
                "Test samples": test_images,
                "Train samples": train_images
            }, step=step)

        print(f"=========== Starting Saving {epoch} ===========")
        save_trainer(save_path, trainer, epoch, step)

    
def main(
    cuda=True,
    epochs=20,
    epoch_length=None,
    validation_length=None,
    n_samples=5,
    do_overfit=False,
    save_path="./models",
    learning_rate=1.2e-4,
    weight_decay=0,
    batch_size=32,
    img_size=(64, 64),
    use_ema=True,
    start_shard = 169230,
    end_shard = 169400,
):
    # Create the save path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cuda = cuda and torch.cuda.is_available()
    if cuda:
        device = torch.device("cuda:7")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    all_shards = list(range(start_shard, end_shard+1))

    dataloaders = create_dataloaders (
        available_shards=all_shards,
        webdataset_base_url="pipe:s3cmd get s3://laion-us-east-1/laion-data/laion2B-data/{}.tar -",
        embeddings_url="s3://dalle2-training-dataset/new_shard_width/reordered_embeddings/",
        num_workers=4,
        batch_size=batch_size,
        shuffle_train=False,
        shuffle_val_test=False,
        img_size=img_size,
        img_preproc = None,
        index_width=4,
        train_prop = 0.75,
        val_prop = 0.15,
        test_prop = 0.10
    )

    decoder, (unet_config, decoder_config) = create_decoder(device)

    wandb.init(
        entity="Veldrovive",
        project="dalle2_train_decoder",
        config={
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "img_size": img_size,
            "use_ema": use_ema,
            "start_shard": start_shard,
            "end_shard": end_shard,
            "unet_config": unet_config,
            "decoder_config": decoder_config
        }
    )

    if not do_overfit:
        train(dataloaders, decoder,
            device=device,
            epochs=epochs,
            epoch_length=epoch_length,
            validation_length=validation_length,
            n_samples=n_samples,
            save_path=save_path,
            use_ema=use_ema,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        overfit(
            dataloaders["val"],
            decoder,
            epoch_length=epoch_length,
            epochs=epochs,
            n_samples=n_samples,
            device=device,
            use_ema=use_ema
        )

if __name__ == "__main__":
    # main(False, epoch_length=None, validation_length=None, n_samples=5, learning_rate=4e-3)
    # main(False, epoch_length=10000, n_samples=3, do_overfit=True)

    all_shards = list(range(169230, 169400+1))
    dataloaders = create_dataloaders (
        available_shards=all_shards,
        webdataset_base_url="pipe:s3cmd get s3://laion-us-east-1/laion-data/laion2B-data/{}.tar -",
        embeddings_url="s3://dalle2-training-dataset/new_shard_width/reordered_embeddings/",
        num_workers=4,
        batch_size=32,
        shuffle_train=False,
        shuffle_val_test=False,
        img_size=(256, 256),
        img_preproc = T.ToTensor(),
        index_width=4,
        train_prop = 0.75,
        val_prop = 0.15,
        test_prop = 0.10
    )

    import clip
    import torchvision.transforms as T
    import numpy as np
    transform = T.ToPILImage()
    device = torch.device("cpu")
    model, preprocess = clip.load("ViT-L/14", device=device)
    skip = 10
    with torch.no_grad(), open("./failed_embeddings.txt", "w") as f, open("./success_embeddings.txt", "w") as g:
        print("Starting testing")
        sample = 0
        for imgs, embs, txts in dataloaders["test"]:
            for img, emb, txt in zip(imgs, embs, txts):
                sample += 1
                if skip > 0 and sample % skip != 0:
                    continue
                try:
                    image = preprocess(transform(img)).unsqueeze(0).to(device)
                    text = clip.tokenize(txt).to(device)
                    image_features = model.encode_image(image).squeeze(0)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    image_features = image_features.detach().cpu().numpy()
                    text_features = model.encode_text(text).squeeze(0)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_features = text_features.detach().cpu().numpy()
                    emb /= emb.norm(dim=-1, keepdim=True)
                    emb = emb.detach().cpu().numpy()
                    img_alignment = np.dot(image_features, emb)
                    txt_alignment = np.dot(text_features, emb)
                    print(f"{sample} {img_alignment} {txt_alignment}")
                    if img_alignment < 0.9:
                        print("Image alignment low:", img_alignment)
                        # Write a line with the sample index and the alignment
                        f.write(f"{sample}\t{img_alignment}\t{txt_alignment}\n")
                        f.flush()
                    else:
                        g.write(f"{sample}\t{img_alignment}\t{txt_alignment}\n")
                        g.flush()
                except Exception as e:
                    print(e)
