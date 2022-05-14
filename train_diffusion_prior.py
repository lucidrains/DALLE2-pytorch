import os
import math
import argparse
import numpy as np

import torch
import clip
from torch import nn
from embedding_reader import EmbeddingReader
from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork, OpenAIClipAdapter
from dalle2_pytorch.train import load_diffusion_model, save_diffusion_model, print_ribbon
from dalle2_pytorch.optimizer import get_optimizer
from torch.cuda.amp import autocast, GradScaler

import time
from tqdm import tqdm

import wandb
os.environ["WANDB_SILENT"] = "true"
NUM_TEST_EMBEDDINGS = 100 # for cosine similarity reporting during training
REPORT_METRICS_EVERY = 100 # for cosine similarity and other metric reporting during training

def caption_to_tokens(captions):
    return clip.tokenize(captions['caption'].to_list(), truncate=True)

def tokens_to_embedding(tokenized_text, diffusion_prior):
    return diffusion_prior.clip.embed_text(tokenized_text)[0]

def eval_model(model,device,image_reader,start,end,batch_size,loss_type,phase="Validation"):
    model.eval()
    with torch.no_grad():
        total_loss = 0.
        total_samples = 0.

        for emb_images, captions in image_reader(batch_size=batch_size, start=start, end=end):
            tokenized_text = caption_to_tokens(captions).to(device)
            emb_text_tensor = tokens_to_embedding(tokenized_text, model)
            emb_images_tensor = torch.tensor(emb_images).to(device)

            batches = emb_images_tensor.shape[0]

            loss = model(text_embed = emb_text_tensor, image_embed = emb_images_tensor)

            total_loss += loss.item() * batches
            total_samples += batches

        avg_loss = (total_loss / total_samples)
        wandb.log({f'{phase} {loss_type}': avg_loss})

def report_cosine_sims(diffusion_prior,image_reader,train_set_size,NUM_TEST_EMBEDDINGS,device):
    diffusion_prior.eval()

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    tstart = train_set_size
    tend = train_set_size+NUM_TEST_EMBEDDINGS

    for embi, captions in image_reader(batch_size=NUM_TEST_EMBEDDINGS, start=tstart, end=tend):
       # tokenize text & embed text
       tokenized_text = caption_to_tokens(captions).to(device)
       text_embed = tokens_to_embedding(tokenized_text, diffusion_prior)

       # make a copy of the text embeddings for shuffling
       text_embed_shuffled = text_embed.clone()
        # roll the text embeddings to simulate "unrelated" captions
       rolled_idx = torch.roll(torch.arange(NUM_TEST_EMBEDDINGS), 1)
       text_embed_shuffled = text_embed_shuffled[rolled_idx]
       text_embed_shuffled = text_embed_shuffled / \
           text_embed_shuffled.norm(dim=1, keepdim=True)
        # prepare the text embedding
       text_embed = text_embed / text_embed.norm(dim=1, keepdim=True)
        # prepare image embeddings
       test_image_embeddings = torch.tensor(embi).to(device)
       test_image_embeddings = test_image_embeddings / \
           test_image_embeddings.norm(dim=1, keepdim=True)
        # predict on the unshuffled text embeddings
       predicted_image_embeddings = diffusion_prior.sample(tokenized_text)
       predicted_image_embeddings = predicted_image_embeddings / \
           predicted_image_embeddings.norm(dim=1, keepdim=True)
        # predict on the shuffled embeddings
       predicted_unrelated_embeddings = diffusion_prior.sample(tokenized_text)
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
       wandb.log({"CosineSimilarity(text_embed,image_embed)": np.mean(original_similarity),
            "CosineSimilarity(text_embed,predicted_image_embed)":np.mean(predicted_similarity),
            "CosineSimilarity(orig_image_embed,predicted_image_embed)":np.mean(predicted_img_similarity),
            "CosineSimilarity(text_embed,predicted_unrelated_embed)": np.mean(unrelated_similarity),
            "Cosine similarity difference":np.mean(predicted_similarity - original_similarity)})

def train(image_embed_dim,
          image_embed_url,
          meta_url,
          batch_size,
          train_percent,
          val_percent,
          test_percent,
          num_epochs,
          dp_loss_type,
          clip_model,
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

    # DiffusionPriorNetwork 
    prior_network = DiffusionPriorNetwork( 
            dim = image_embed_dim, 
            depth = dpn_depth, 
            dim_head = dpn_dim_head, 
            heads = dpn_heads,
            attn_dropout = dropout,
            ff_dropout = dropout,
            normformer = dp_normformer).to(device)
    
    # DiffusionPrior with text embeddings and image embeddings pre-computed
    diffusion_prior = DiffusionPrior( 
            net = prior_network, 
            clip = OpenAIClipAdapter(clip_model),
            image_embed_dim = image_embed_dim, 
            timesteps = dp_timesteps,
            cond_drop_prob = dp_cond_drop_prob, 
            loss_type = dp_loss_type, 
            condition_on_text_encodings = dp_condition_on_text_encodings).to(device)

    # Load pre-trained model from DPRIOR_PATH
    if RESUME:
        diffusion_prior=load_diffusion_model(DPRIOR_PATH,device)   
        wandb.init( entity=wandb_entity, project=wandb_project, config=config) 

    # Create save_path if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Get image and text embeddings from the servers
    print_ribbon("Downloading Image Embeddings")
    image_reader = EmbeddingReader(embeddings_folder=image_embed_url,
                                   metadata_folder=meta_url, meta_columns=['caption'], file_format="parquet_npy")
    num_data_points = image_reader.count

    ### Training code ###
    scaler = GradScaler(enabled=amp)
    optimizer = get_optimizer(diffusion_prior.net.parameters(), wd=weight_decay, lr=learning_rate)
    epochs = num_epochs

    step = 0
    t = time.time()

    train_set_size = int(train_percent*num_data_points)
    val_set_size = int(val_percent*num_data_points)
    eval_start = train_set_size

    for _ in range(epochs):

        for emb_images, captions in image_reader(batch_size=batch_size, start=0, end=train_set_size):

            diffusion_prior.train()

            # tokenize the text & prepare image embeddings
            tokenized_text = caption_to_tokens(captions).to(device)
            emb_images_tensor = torch.tensor(emb_images).to(device)

            with autocast(enabled=amp):
                loss = diffusion_prior(text = tokenized_text, image_embed = emb_images_tensor)
                scaler.scale(loss).backward()

            # Samples per second
            step+=1
            samples_per_sec = batch_size*step/(time.time()-t)
            # Save checkpoint every save_interval minutes
            if(int(time.time()-t) >= 60*save_interval):
                t = time.time()

                save_diffusion_model(
                    save_path,
                    diffusion_prior,
                    optimizer,
                    scaler,
                    config,
                    image_embed_dim)

            # Log to wandb
            wandb.log({"Training loss": loss.item(),
                        "Steps": step,
                        "Samples per second": samples_per_sec})
            # Log cosineSim(text_embed,predicted_image_embed) - cosineSim(text_embed,image_embed)
            # Use NUM_TEST_EMBEDDINGS samples from the test set each time
            # Get embeddings from the most recently saved model
            if(step % REPORT_METRICS_EVERY) == 0:
                report_cosine_sims(diffusion_prior,
                        image_reader,
                        train_set_size,
                        NUM_TEST_EMBEDDINGS,
                        device)
                ### Evaluate model(validation run) ###
                eval_model(diffusion_prior,
                        device,
                        image_reader,
                        eval_start,
                        eval_start+NUM_TEST_EMBEDDINGS,
                        NUM_TEST_EMBEDDINGS,
                        dp_loss_type,
                        phase="Validation")

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(diffusion_prior.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    ### Test run ###
    test_set_size = int(test_percent*train_set_size) 
    start=train_set_size+val_set_size
    end=num_data_points
    eval_model(diffusion_prior,device,image_reader,start,end,batch_size,dp_loss_type,phase="Test")

def main():
    parser = argparse.ArgumentParser()
    # Logging
    parser.add_argument("--wandb-entity", type=str, default="laion")
    parser.add_argument("--wandb-project", type=str, default="diffusion-prior")
    parser.add_argument("--wandb-dataset", type=str, default="LAION-5B")
    parser.add_argument("--wandb-arch", type=str, default="DiffusionPrior")
    # URLs for embeddings 
    parser.add_argument("--image-embed-url", type=str, default="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/img_emb/")
    parser.add_argument("--meta-url", type=str, default="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/laion2B-en-metadata/")
    # Hyperparameters
    parser.add_argument("--learning-rate", type=float, default=1.1e-4)
    parser.add_argument("--weight-decay", type=float, default=6.02e-2)
    parser.add_argument("--dropout", type=float, default=5e-2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=1)
    # Image embed dimension
    parser.add_argument("--image-embed-dim", type=int, default=768)
    # Train-test split
    parser.add_argument("--train-percent", type=float, default=0.7)
    parser.add_argument("--val-percent", type=float, default=0.2)
    parser.add_argument("--test-percent", type=float, default=0.1)
    # DiffusionPriorNetwork(dpn) parameters
    parser.add_argument("--dpn-depth", type=int, default=12)
    parser.add_argument("--dpn-dim-head", type=int, default=64)
    parser.add_argument("--dpn-heads", type=int, default=12)
    # DiffusionPrior(dp) parameters
    parser.add_argument("--dp-condition-on-text-encodings", type=bool, default=True)
    parser.add_argument("--dp-timesteps", type=int, default=1000)
    parser.add_argument("--dp-normformer", type=bool, default=True)
    parser.add_argument("--dp-cond-drop-prob", type=float, default=0.1)
    parser.add_argument("--dp-loss-type", type=str, default="l2")
    parser.add_argument("--clip", type=str, default="ViT-L/14")
    parser.add_argument("--amp", type=bool, default=False)
    # Model checkpointing interval(minutes)
    parser.add_argument("--save-interval", type=int, default=120)
    parser.add_argument("--save-path", type=str, default="./diffusion_prior_checkpoints")
    # Saved model path 
    parser.add_argument("--pretrained-model-path", type=str, default=None)
    # GPU selection
    parser.add_argument("--gpu-device", type=int, default=0)

    args = parser.parse_args()

    config = ({"learning_rate": args.learning_rate,
        "architecture": args.wandb_arch,
        "dataset": args.wandb_dataset,
        "weight_decay":args.weight_decay,
        "max_gradient_clipping_norm":args.max_grad_norm,
        "batch_size":args.batch_size,
        "epochs": args.num_epochs,
        "diffusion_prior_network":{"depth":args.dpn_depth,
        "dim_head":args.dpn_dim_head,
        "heads":args.dpn_heads,
        "normformer":args.dp_normformer},
        "diffusion_prior":{"condition_on_text_encodings": args.dp_condition_on_text_encodings,
        "timesteps": args.dp_timesteps,
        "cond_drop_prob":args.dp_cond_drop_prob,
        "loss_type":args.dp_loss_type,
        "clip":args.clip}
        })

    RESUME = False
    # Check if DPRIOR_PATH exists(saved model path)
    DPRIOR_PATH = args.pretrained_model_path
    if(DPRIOR_PATH is not None):
        RESUME = True
    else:
        wandb.init(
          entity=args.wandb_entity,
          project=args.wandb_project,
          config=config)

    # Obtain the utilized device.

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        device = torch.device(f"cuda:{args.gpu_device}")
        torch.cuda.set_device(device)

    # Training loop
    train(args.image_embed_dim,
          args.image_embed_url,
          args.meta_url,
          args.batch_size,
          args.train_percent,
          args.val_percent,
          args.test_percent,
          args.num_epochs,
          args.dp_loss_type,
          args.clip,
          args.dp_condition_on_text_encodings,
          args.dp_timesteps,
          args.dp_normformer,
          args.dp_cond_drop_prob,
          args.dpn_depth,
          args.dpn_dim_head,
          args.dpn_heads,
          args.save_interval,
          args.save_path,
          device,
          RESUME,
          DPRIOR_PATH,
          config,
          args.wandb_entity,
          args.wandb_project,
          args.learning_rate,
          args.max_grad_norm,
          args.weight_decay,
          args.dropout,
          args.amp)

if __name__ == "__main__":
  main()
