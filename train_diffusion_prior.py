from pathlib import Path
import click
import math
import time
import numpy as np

import torch
from torch import nn

from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork
from dalle2_pytorch.train import DiffusionPriorTrainer, load_diffusion_model, save_diffusion_model, print_ribbon
from dalle2_pytorch.trackers import ConsoleTracker, WandbTracker

from embedding_reader import EmbeddingReader

from tqdm import tqdm

# constants

NUM_TEST_EMBEDDINGS = 100 # for cosine similarity reporting during training
REPORT_METRICS_EVERY = 100 # for cosine similarity and other metric reporting during training

tracker = WandbTracker()

# helpers functions

class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_time = time.time()

    def elapsed(self):
        return time.time() - self.last_time
# functions

def eval_model(model,device,image_reader,text_reader,start,end,batch_size,loss_type,phase="Validation"):
    model.eval()
    with torch.no_grad():
        total_loss = 0.
        total_samples = 0.

        for emb_images, emb_text in zip(image_reader(batch_size=batch_size, start=start, end=end),
                text_reader(batch_size=batch_size, start=start, end=end)):

            emb_images_tensor = torch.tensor(emb_images[0]).to(device)
            emb_text_tensor = torch.tensor(emb_text[0]).to(device)

            batches = emb_images_tensor.shape[0]

            loss = model(text_embed = emb_text_tensor, image_embed = emb_images_tensor)

            total_loss += loss.item() * batches
            total_samples += batches

        avg_loss = (total_loss / total_samples)
        tracker.log({f'{phase} {loss_type}': avg_loss})

def report_cosine_sims(diffusion_prior,image_reader,text_reader,train_set_size,NUM_TEST_EMBEDDINGS,device):
    diffusion_prior.eval()

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    tstart = train_set_size
    tend = train_set_size+NUM_TEST_EMBEDDINGS

    for embt, embi in zip(text_reader(batch_size=NUM_TEST_EMBEDDINGS, start=tstart, end=tend), 
            image_reader(batch_size=NUM_TEST_EMBEDDINGS, start=tstart, end=tend)):
       # make a copy of the text embeddings for shuffling
       text_embed = torch.tensor(embt[0]).to(device)
       text_embed_shuffled = text_embed.clone()
        # roll the text embeddings to simulate "unrelated" captions
       rolled_idx = torch.roll(torch.arange(NUM_TEST_EMBEDDINGS), 1)
       text_embed_shuffled = text_embed_shuffled[rolled_idx]
       text_embed_shuffled = text_embed_shuffled / \
           text_embed_shuffled.norm(dim=1, keepdim=True)
       test_text_shuffled_cond = dict(text_embed=text_embed_shuffled)
        # prepare the text embedding
       text_embed = text_embed / text_embed.norm(dim=1, keepdim=True)
       test_text_cond = dict(text_embed=text_embed)
        # prepare image embeddings
       test_image_embeddings = torch.tensor(embi[0]).to(device)
       test_image_embeddings = test_image_embeddings / \
           test_image_embeddings.norm(dim=1, keepdim=True)
        # predict on the unshuffled text embeddings
       predicted_image_embeddings = diffusion_prior.p_sample_loop(
           (NUM_TEST_EMBEDDINGS, 768), text_cond=test_text_cond)
       predicted_image_embeddings = predicted_image_embeddings / \
           predicted_image_embeddings.norm(dim=1, keepdim=True)
        # predict on the shuffled embeddings
       predicted_unrelated_embeddings = diffusion_prior.p_sample_loop(
           (NUM_TEST_EMBEDDINGS, 768), text_cond=test_text_shuffled_cond)
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

def train(image_embed_dim,
          image_embed_url,
          text_embed_url,
          batch_size,
          train_percent,
          val_percent,
          test_percent,
          num_epochs,
          dp_loss_type,
          clip,
          dp_condition_on_text_encodings,
          dp_timesteps,
          dp_normformer,
          dp_cond_drop_prob,
          dpn_depth,
          dpn_dim_head,
          dpn_heads,
          save_interval,
          save_path,
          device,
          RESUME,
          DPRIOR_PATH,
          config,
          wandb_entity,
          wandb_project,
          learning_rate=0.001,
          max_grad_norm=0.5,
          weight_decay=0.01,
          dropout=0.05,
          amp=False):

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
    
    # diffusion prior with text embeddings and image embeddings pre-computed

    diffusion_prior = DiffusionPrior( 
        net = prior_network,
        clip = clip,
        image_embed_dim = image_embed_dim,
        timesteps = dp_timesteps,
        cond_drop_prob = dp_cond_drop_prob,
        loss_type = dp_loss_type,
        condition_on_text_encodings = dp_condition_on_text_encodings
    )

    # Load pre-trained model from DPRIOR_PATH

    if RESUME:
        diffusion_prior, loaded_obj = load_diffusion_model(DPRIOR_PATH, device)

        # TODO, optimizer and scaler needs to be loaded as well

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

    # Get image and text embeddings from the servers

    print_ribbon("Downloading embeddings - image and text")
    image_reader = EmbeddingReader(embeddings_folder=image_embed_url, file_format="npy")
    text_reader  = EmbeddingReader(embeddings_folder=text_embed_url, file_format="npy")
    num_data_points = text_reader.count

    ### Training code ###

    timer = Timer()
    epochs = num_epochs

    train_set_size = int(train_percent*num_data_points)
    val_set_size = int(val_percent*num_data_points)
    eval_start = train_set_size

    for _ in range(epochs):

        for emb_images,emb_text in zip(image_reader(batch_size=batch_size, start=0, end=train_set_size),
                text_reader(batch_size=batch_size, start=0, end=train_set_size)):

            trainer.train()
            
            emb_images_tensor = torch.tensor(emb_images[0]).to(device)
            emb_text_tensor = torch.tensor(emb_text[0]).to(device)

            loss = trainer(text_embed = emb_text_tensor, image_embed = emb_images_tensor)

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
            tracker.log({"Training loss": loss.item(),
                        "Steps": step,
                        "Samples per second": samples_per_sec})
            # Log cosineSim(text_embed,predicted_image_embed) - cosineSim(text_embed,image_embed)
            # Use NUM_TEST_EMBEDDINGS samples from the test set each time
            # Get embeddings from the most recently saved model
            if(step % REPORT_METRICS_EVERY) == 0:
                report_cosine_sims(diffusion_prior,
                        image_reader,
                        text_reader,
                        train_set_size,
                        NUM_TEST_EMBEDDINGS,
                        device)
                ### Evaluate model(validation run) ###
                eval_model(diffusion_prior,
                        device,
                        image_reader,
                        text_reader,
                        eval_start,
                        eval_start+NUM_TEST_EMBEDDINGS,
                        NUM_TEST_EMBEDDINGS,
                        dp_loss_type,
                        phase="Validation")

            trainer.update()

    ### Test run ###
    test_set_size = int(test_percent*train_set_size) 
    start = train_set_size+val_set_size
    end = num_data_points
    eval_model(diffusion_prior,device,image_reader,text_reader,start,end,batch_size,dp_loss_type,phase="Test")

@click.command()
@click.option("--wandb-entity", default="laion")
@click.option("--wandb-project", default="diffusion-prior")
@click.option("--wandb-dataset", default="LAION-5B")
@click.option("--wandb-arch", default="DiffusionPrior")
@click.option("--image-embed-url", default="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/img_emb/")
@click.option("--text-embed-url", default="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/text_emb/")
@click.option("--learning-rate", default=1.1e-4)
@click.option("--weight-decay", default=6.02e-2)
@click.option("--dropout", default=5e-2)
@click.option("--max-grad-norm", default=0.5)
@click.option("--batch-size", default=10**4)
@click.option("--num-epochs", default=5)
@click.option("--image-embed-dim", default=768)
@click.option("--train-percent", default=0.7)
@click.option("--val-percent", default=0.2)
@click.option("--test-percent", default=0.1)
@click.option("--dpn-depth", default=6)
@click.option("--dpn-dim-head", default=64)
@click.option("--dpn-heads", default=8)
@click.option("--dp-condition-on-text-encodings", default=False)
@click.option("--dp-timesteps", default=100)
@click.option("--dp-normformer", default=False)
@click.option("--dp-cond-drop-prob", default=0.1)
@click.option("--dp-loss-type", default="l2")
@click.option("--clip", default=None)
@click.option("--amp", default=False)
@click.option("--save-interval", default=30)
@click.option("--save-path", default="./diffusion_prior_checkpoints")
@click.option("--pretrained-model-path", default=None)
def main(
    wandb_entity,
    wandb_project,
    wandb_dataset,
    wandb_arch,
    image_embed_url,
    text_embed_url,
    learning_rate,
    weight_decay,
    dropout,
    max_grad_norm,
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
    pretrained_model_path
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

    RESUME = False

    # Check if DPRIOR_PATH exists(saved model path)

    DPRIOR_PATH = args.pretrained_model_path

    if(DPRIOR_PATH is not None):
        RESUME = True
    else:
        tracker.init(
            entity = wandb_entity,
            project = wandb_project,
            config = config
        )

    # Obtain the utilized device.

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # Training loop
    train(image_embed_dim,
          image_embed_url,
          text_embed_url,
          batch_size,
          train_percent,
          val_percent,
          test_percent,
          num_epochs,
          dp_loss_type,
          clip,
          dp_condition_on_text_encodings,
          dp_timesteps,
          dp_normformer,
          dp_cond_drop_prob,
          dpn_depth,
          dpn_dim_head,
          dpn_heads,
          save_interval,
          save_path,
          device,
          RESUME,
          DPRIOR_PATH,
          config,
          wandb_entity,
          wandb_project,
          learning_rate,
          max_grad_norm,
          weight_decay,
          dropout,
          amp)

if __name__ == "__main__":
    main()
