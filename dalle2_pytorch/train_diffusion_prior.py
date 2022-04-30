import argparse
import os
from dalle2_pytorch import DiffusionPrior
from embedding_reader import EmbeddingReader
from dalle2_pytorch import DiffusionPriorNetwork
import numpy as np
import math
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import AdamW
import wandb
os.environ["WANDB_SILENT"] = "true"

def separate_weight_decayable_params(params):
    no_wd_params = set([param for param in params if param.ndim < 2])
    wd_params = set(params) - no_wd_params
    return wd_params, no_wd_params

def get_optimizer(params, lr = 3e-4, wd = 1e-1, filter_by_requires_grad = False):
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))

    params = set(params)
    wd_params, no_wd_params = separate_weight_decayable_params(params)

    param_groups = [
        {'params': list(wd_params)},
        {'params': list(no_wd_params), 'weight_decay': 0},
    ]

    return AdamW(param_groups, lr = lr, weight_decay = wd)

def eval_model(model,device,image_reader,text_reader,start,end,batch_size,isValidation=True):
    with torch.no_grad():
        for emb_images,emb_text in zip(image_reader(batch_size=batch_size, start=start, end=end),
                text_reader(batch_size=batch_size, start=start, end=end)):
            emb_images = list(emb_images)
            emb_text = list(emb_text)
            emb_images[0] = torch.tensor(emb_images[0]).to(device)
            emb_text[0] = torch.tensor(emb_text[0]).to(device)
            model.eval()
            loss = model(text_embed = emb_text[0],image_embed = emb_images[0])

            # Log to wandb
            if(isValidation):
                wandb.log({"Validation mse": loss})
            else:
                wandb.log({"Test mse": loss})

    # Saving State Dict
    torch.save(model.state_dict(), 'saved_model.pth')

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
          dp_cond_drop_prob,
          dpn_depth,
          dpn_dim_head,
          dpn_heads,
          device,
          learning_rate=0.01):


    # DiffusionPriorNetwork 
    prior_network = DiffusionPriorNetwork( 
            dim = image_embed_dim, 
            depth = dpn_depth, 
            dim_head = dpn_dim_head, 
            heads = dpn_heads).to(device)
    
    # DiffusionPrior with text embeddings and image embeddings pre-computed
    diffusion_prior = DiffusionPrior( 
            net = prior_network, 
            clip = clip, 
            image_embed_dim = image_embed_dim, 
            timesteps = dp_timesteps,
            cond_drop_prob = dp_cond_drop_prob, 
            loss_type = dp_loss_type, 
            condition_on_text_encodings = dp_condition_on_text_encodings).to(device)

    # Get image and text embeddings from the servers
    print("==============Downloading embeddings - image and text====================")
    image_reader = EmbeddingReader(embeddings_folder=image_embed_url, file_format="npy")
    text_reader = EmbeddingReader(embeddings_folder=text_embed_url, file_format="npy")
    num_data_points = text_reader.count

    ### Training code ###
    optimizer = get_optimizer(diffusion_prior.parameters())
    epochs = num_epochs

    for _ in range(epochs):
        train_set_size = int(train_percent*num_data_points)
        for emb_images,emb_text in zip(image_reader(batch_size=batch_size, start=0, end=train_set_size),
                text_reader(batch_size=batch_size, start=0, end=train_set_size)):
            emb_images = list(emb_images)
            emb_text = list(emb_text)
            emb_images[0] = torch.tensor(emb_images[0]).to(device)
            emb_text[0] = torch.tensor(emb_text[0]).to(device)
            optimizer.zero_grad()
            loss = diffusion_prior(text_embed = emb_text[0],image_embed = emb_images[0])
            loss.backward()
            # Log to wandb
            wandb.log({"Training (mse)": loss})
            optimizer.step()
            print("Training mse = ",loss.item())

        ### Evaluate model(validation run) ###
        val_set_size = int(val_percent*num_data_points)
        start = train_set_size
        end=start+val_set_size
        eval_model(diffusion_prior,device,image_reader,text_reader,start,end,batch_size,isValidation=True)

    ## Test run ###
    test_set_size = int(test_percent*train_set_size) 
    start=train_set_size+val_set_size
    end=num_data_points
    eval_model(diffusion_prior,device,image_reader,text_reader,start,end,batch_size,isValidation=False)

def main():
    parser = argparse.ArgumentParser()
    # Logging
    parser.add_argument("--wandb-entity", type=str, default="laion")
    parser.add_argument("--wandb-project", type=str, default="diffusion-prior")
    parser.add_argument("--wandb-name", type=str, default="laion-dprior")
    parser.add_argument("--wandb-dataset", type=str, default="LAION-5B")
    parser.add_argument("--wandb-arch", type=str, default="DiffusionPrior")



    # URLs for embeddings 
    parser.add_argument("--image-embed-url", type=str, default="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/img_emb/")
    parser.add_argument("--text-embed-url", type=str, default="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/text_emb/")
    # Hyperparameters
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=10**4)
    parser.add_argument("--num-epochs", type=int, default=5)
    # Image embed dimension
    parser.add_argument("--image-embed-dim", type=int, default=768)
    # Train-test split
    parser.add_argument("--train-percent", type=float, default=0.7)
    parser.add_argument("--val-percent", type=float, default=0.2)
    parser.add_argument("--test-percent", type=float, default=0.1)
    # LAION training(pre-computed embeddings)
    # DiffusionPriorNetwork(dpn) parameters
    parser.add_argument("--dpn-depth", type=int, default=6)
    parser.add_argument("--dpn-dim-head", type=int, default=64)
    parser.add_argument("--dpn-heads", type=int, default=8)
    # DiffusionPrior(dp) parameters
    parser.add_argument("--dp-condition-on-text-encodings", type=bool, default=False)
    parser.add_argument("--dp-timesteps", type=int, default=100)
    parser.add_argument("--dp-cond-drop-prob", type=float, default=0.2)
    parser.add_argument("--dp-loss-type", type=str, default="l2")
    parser.add_argument("--clip", type=str, default=None)

    args = parser.parse_args()
    print("Setting up wandb logging... Please wait...")
    wandb.init(
      entity=args.wandb_entity,
      project=args.wandb_project,
      name=args.wandb_name,
      config={
      "learning_rate": args.learning_rate,
      "architecture": args.wandb_arch,
      "dataset": args.wandb_dataset,
      "epochs": args.num_epochs,
      })
    print("wandb logging setup done!")
    # Obtain the utilized device.
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        has_cuda = True
    else:
        device = torch.device("cpu")
        has_cuda = False
    # Training loop
    train(args.image_embed_dim,
          args.image_embed_url,
          args.text_embed_url,
          args.batch_size,
          args.train_percent,
          args.val_percent,
          args.test_percent,
          args.num_epochs,
          args.dp_loss_type,
          args.clip,
          args.dp_condition_on_text_encodings,
          args.dp_timesteps,
          args.dp_cond_drop_prob,
          args.dpn_depth,
          args.dpn_dim_head,
          args.dpn_heads,
          device,
          args.learning_rate)


if __name__ == "__main__":
  main()

