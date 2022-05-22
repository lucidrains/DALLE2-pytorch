from pathlib import Path
import click
import math
import numpy as np

import torch
import clip
from torch import nn

from dalle2_pytorch.dataloaders import make_splits
from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork, OpenAIClipAdapter
from dalle2_pytorch.trainer import DiffusionPriorTrainer, load_diffusion_model, save_diffusion_model

from dalle2_pytorch.trackers import ConsoleTracker, WandbTracker
from dalle2_pytorch.utils import Timer, print_ribbon

from embedding_reader import EmbeddingReader

from tqdm import tqdm

# constants

REPORT_METRICS_EVERY = 250 # for cosine similarity and other metric reporting during training

tracker = WandbTracker()

# helpers functions

def exists(val):
    val is not None

# functions

def eval_model(model, dataloader, text_conditioned, loss_type, phase="Validation"):
    model.eval()

    with torch.no_grad():
        total_loss = 0.
        total_samples = 0.

        for image_embeddings, text_data in tqdm(dataloader):

            batches = image_embeddings.shape[0]

            input_args = dict(image_embed=image_embeddings)
            if text_conditioned:
                input_args = dict(**input_args, text = text_data)
            else:
                input_args = dict(**input_args, text_embed=text_data)

            loss = model(**input_args)

            total_loss += loss * batches
            total_samples += batches

        avg_loss = (total_loss / total_samples)

        tracker.log({f'{phase} {loss_type}': avg_loss})

def report_cosine_sims(diffusion_prior, dataloader, text_conditioned):
    diffusion_prior.eval()

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    for test_image_embeddings, text_data in tqdm(dataloader):

        # we are text conditioned, we produce an embedding from the tokenized text
        if text_conditioned:
            text_embedding, text_encodings, text_mask = diffusion_prior.clip.embed_text(
                text_data)
            text_cond = dict(text_embed=text_embedding,
                             text_encodings=text_encodings, mask=text_mask)
        else:
            text_embedding = text_data
            text_cond = dict(text_embed=text_embedding)

        # make a copy of the text embeddings for shuffling
        text_embed_shuffled = text_embedding.clone()

        # roll the text to simulate "unrelated" captions
        rolled_idx = torch.roll(torch.arange(text_embedding.shape[0]), 1)
        text_embed_shuffled = text_embed_shuffled[rolled_idx]
        text_embed_shuffled = text_embed_shuffled / \
            text_embed_shuffled.norm(dim=1, keepdim=True)

        if text_conditioned:
            text_encodings_shuffled = text_encodings[rolled_idx]
            text_mask_shuffled = text_mask[rolled_idx]
        else:
            text_encodings_shuffled = None
            text_mask_shuffled = None

        text_cond_shuffled = dict(text_embed=text_embed_shuffled,
                                  text_encodings=text_encodings_shuffled, mask=text_mask_shuffled)

        # prepare the text embedding
        text_embed = text_embedding / text_embedding.norm(dim=1, keepdim=True)

        # prepare image embeddings
        test_image_embeddings = test_image_embeddings / \
            test_image_embeddings.norm(dim=1, keepdim=True)

        # predict on the unshuffled text embeddings
        predicted_image_embeddings = diffusion_prior.p_sample_loop(
            test_image_embeddings.shape, text_cond)
        predicted_image_embeddings = predicted_image_embeddings / \
            predicted_image_embeddings.norm(dim=1, keepdim=True)

        # predict on the shuffled embeddings
        predicted_unrelated_embeddings = diffusion_prior.p_sample_loop(
            test_image_embeddings.shape, text_cond_shuffled)
        predicted_unrelated_embeddings = predicted_unrelated_embeddings / \
            predicted_unrelated_embeddings.norm(dim=1, keepdim=True)

        # calculate similarities
        original_similarity = cos(
           text_embed, test_image_embeddings).cpu().numpy()
        predicted_similarity = cos(
           text_embed, predicted_image_embeddings).cpu().numpy()
        unrelated_similarity = cos(
           text_embed, predicted_unrelated_embeddings).cpu().numpy()
        predicted_img_similarity = cos(
           test_image_embeddings, predicted_image_embeddings).cpu().numpy()
        tracker.log({"CosineSimilarity(text_embed,image_embed)": np.mean(original_similarity),
            "CosineSimilarity(text_embed,predicted_image_embed)":np.mean(predicted_similarity),
            "CosineSimilarity(orig_image_embed,predicted_image_embed)":np.mean(predicted_img_similarity),
            "CosineSimilarity(text_embed,predicted_unrelated_embed)": np.mean(unrelated_similarity),
            "Cosine similarity difference":np.mean(predicted_similarity - original_similarity)})


@click.command()
@click.option("--wandb-entity", default="laion")
@click.option("--wandb-project", default="diffusion-prior")
@click.option("--wandb-dataset", default="LAION-5B")
@click.option("--wandb-arch", default="DiffusionPrior")
@click.option("--image-embed-url", default="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/img_emb/")
@click.option("--text-embed-url", default="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/text_emb/")
@click.option("--meta-url", default="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/laion2B-en-metadata/")
@click.option("--learning-rate", default=1.1e-4)
@click.option("--weight-decay", default=6.02e-2)
@click.option("--dropout", default=5e-2)
@click.option("--max-grad-norm", default=0.5)
@click.option("--num-data-points", default=250e6)
@click.option("--batch-size", default=320)
@click.option("--num-epochs", default=5)
@click.option("--image-embed-dim", default=768)
@click.option("--train-percent", default=0.9)
@click.option("--val-percent", default=1e-7)
@click.option("--test-percent", default=0.0999999)
@click.option("--dpn-depth", default=12)
@click.option("--dpn-dim-head", default=64)
@click.option("--dpn-heads", default=12)
@click.option("--dp-condition-on-text-encodings", default=True)
@click.option("--dp-timesteps", default=1000)
@click.option("--dp-normformer", default=True)
@click.option("--dp-cond-drop-prob", default=0.1)
@click.option("--dp-loss-type", default="l2")
@click.option("--clip", default="ViT-L/14")
@click.option("--amp", default=False)
@click.option("--save-interval", default=120)
@click.option("--save-path", default="./diffusion_prior_checkpoints")
@click.option("--pretrained-model-path", default=None)
@click.option("--gpu-device", default=0)
def train(
    wandb_entity,
    wandb_project,
    wandb_dataset,
    wandb_arch,
    image_embed_url,
    text_embed_url,
    meta_url,
    learning_rate,
    weight_decay,
    dropout,
    max_grad_norm,
    num_data_points,
    batch_size,
    num_epochs,
    image_embed_dim,
    train_percent,
    val_percent,
    test_percent,
    dpn_depth,
    dpn_dim_head,
    dpn_heads,
    dp_condition_on_text_encodings,
    dp_timesteps,
    dp_normformer,
    dp_cond_drop_prob,
    dp_loss_type,
    clip,
    amp,
    save_interval,
    save_path,
    pretrained_model_path,
    gpu_device
):
    config = {
        "learning_rate": learning_rate,
        "architecture": wandb_arch,
        "dataset": wandb_dataset,
        "weight_decay": weight_decay,
        "max_gradient_clipping_norm": max_grad_norm,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "diffusion_prior_network": {
            "depth": dpn_depth,
            "dim_head": dpn_dim_head,
            "heads": dpn_heads,
            "normformer": dp_normformer
        },
        "diffusion_prior": {
            "condition_on_text_encodings": dp_condition_on_text_encodings,
            "timesteps": dp_timesteps,
            "cond_drop_prob": dp_cond_drop_prob,
            "loss_type": dp_loss_type,
            "clip": clip
        }
    }

    # Check if DPRIOR_PATH exists(saved model path)

    DPRIOR_PATH = pretrained_model_path
    RESUME = exists(DPRIOR_PATH)

    if not RESUME:
        tracker.init(
            entity = wandb_entity,
            project = wandb_project,
            config = config
        )

    # Obtain the utilized device.

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        device = torch.device(f"cuda:{gpu_device}")
        torch.cuda.set_device(device)

    # Training loop
    # diffusion prior network

    prior_network = DiffusionPriorNetwork( 
        dim = image_embed_dim,
        depth = dpn_depth,
        dim_head = dpn_dim_head,
        heads = dpn_heads,
        attn_dropout = dropout,
        ff_dropout = dropout,
        normformer = dp_normformer
    )
    
    # Load clip model if text-conditioning
    if dp_condition_on_text_encodings:
        clip_adapter = OpenAIClipAdapter(clip)
    else:
        clip_adapter = None
        
    # diffusion prior with text embeddings and image embeddings pre-computed

    diffusion_prior = DiffusionPrior( 
        net = prior_network,
        clip = clip_adapter,
        image_embed_dim = image_embed_dim,
        timesteps = dp_timesteps,
        cond_drop_prob = dp_cond_drop_prob,
        loss_type = dp_loss_type,
        condition_on_text_encodings = dp_condition_on_text_encodings
    )

    # Load pre-trained model from DPRIOR_PATH

    if RESUME:
        diffusion_prior, loaded_obj = load_diffusion_model(DPRIOR_PATH, device)
        tracker.init(entity = wandb_entity, project = wandb_project, config = config)

    # diffusion prior trainer

    trainer = DiffusionPriorTrainer(
        diffusion_prior = diffusion_prior,
        lr = learning_rate,
        wd = weight_decay,
        max_grad_norm = max_grad_norm,
        amp = amp,
    ).to(device)

    # load optimizer and scaler

    if RESUME:
        trainer.optimizer.load_state_dict(loaded_obj['optimizer'])
        trainer.scaler.load_state_dict(loaded_obj['scaler'])

    # Create save_path if it doesn't exist

    Path(save_path).mkdir(exist_ok = True, parents = True)

    # Utilize wrapper to abstract away loader logic
    print_ribbon("Downloading Embeddings")
    loader_args = dict(text_conditioned=dp_condition_on_text_encodings, batch_size=batch_size, num_data_points=num_data_points,
                       train_split=train_percent, eval_split=val_percent, device=device, img_url=image_embed_url)

    if dp_condition_on_text_encodings:
        loader_args = dict(**loader_args, meta_url=meta_url)
    else:
        loader_args = dict(**loader_args, txt_url=text_embed_url)

    train_loader, eval_loader, test_loader = make_splits(**loader_args)

    ### Training code ###

    step = 1 
    timer = Timer()
    epochs = num_epochs

    for _ in range(epochs):

        for image, text in tqdm(train_loader):
            
            diffusion_prior.train()
            
            input_args = dict(image_embed=image)
            if dp_condition_on_text_encodings:
                input_args = dict(**input_args, text = text)
            else:
                input_args = dict(**input_args, text_embed=text)

            loss = trainer(**input_args)

            # Samples per second

            samples_per_sec = batch_size * step / timer.elapsed()

            # Save checkpoint every save_interval minutes
            if(int(timer.elapsed()) >= 60 * save_interval):
                timer.reset()

                save_diffusion_model(
                    save_path,
                    diffusion_prior,
                    trainer.optimizer,
                    trainer.scaler,
                    config,
                    image_embed_dim)

            # Log to wandb
            tracker.log({"Training loss": loss,
                        "Steps": step,
                        "Samples per second": samples_per_sec})
            # Log cosineSim(text_embed,predicted_image_embed) - cosineSim(text_embed,image_embed)
            # Use NUM_TEST_EMBEDDINGS samples from the test set each time
            # Get embeddings from the most recently saved model
            if(step % REPORT_METRICS_EVERY) == 0:
                report_cosine_sims(diffusion_prior, eval_loader, dp_condition_on_text_encodings)
                ### Evaluate model(validation run) ###
                eval_model(diffusion_prior, eval_loader, dp_condition_on_text_encodings, dp_loss_type, phase="Validation")

            step += 1
            trainer.update()

    ### Test run ###
    eval_model(diffusion_prior, test_loader, dp_condition_on_text_encodings, dp_loss_type, phase="Test")


if __name__ == "__main__":
    train()
