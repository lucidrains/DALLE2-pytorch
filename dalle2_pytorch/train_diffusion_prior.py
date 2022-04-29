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
import wandb
os.environ["WANDB_SILENT"] = "true"

def train(image_embed_dim,
          image_embed_url,
          text_embed_url,
          batch_size,
          train_percent,
          val_percent,test_percent,
          num_epochs,
          loss_type,
          condition_on_text_encodings,
          device,
          learning_rate=0.01):
    # DiffusionPriorNetwork 
    prior_network = DiffusionPriorNetwork( dim = image_embed_dim, depth = 6, dim_head = 64, heads = 8).to(device)
    
    # DiffusionPrior with text embeddings and image embeddings pre-computed
    diffusion_prior = DiffusionPrior( net = prior_network, clip = None, image_embed_dim = image_embed_dim, 
                                     timesteps = 100, cond_drop_prob = 0.2, loss_type = loss_type,
                                     condition_on_text_encodings = condition_on_text_encodings).to(device)
    # Gtext_reader image and text embeddings from the servers
    print("==============Downloading embeddings - image and text====================")
    image_reader = EmbeddingReader(embeddings_folder=image_embed_url, file_format="npy")
    text_reader = EmbeddingReader(embeddings_folder=text_embed_url, file_format="npy")

    ### Training code ###
    optimizer = torch.optim.SGD(diffusion_prior.parameters(), lr = learning_rate)
    epochs = num_epochs
    min_valid_loss = np.inf
    for e in range(epochs):
        train_loss = 0.0 
        print("Training loop - epoch number ",e)
        train_set_size = int(train_percent*image_reader.count)
        for emb_images,emb_text in zip(image_reader(batch_size=batch_size, start=0, end=train_set_size),
                text_reader(batch_size=batch_size, start=0, end=train_set_size)):
            emb_images = list(emb_images)
            emb_text = list(emb_text)
            print(emb_images[0].shape,emb_text[0].shape)
            emb_images[0] = torch.tensor(emb_images[0]).to(device)
            emb_text[0] = torch.tensor(emb_text[0]).to(device)
            optimizer.zero_grad()
            loss = diffusion_prior(text_embed = emb_text[0],image_embed = emb_images[0])
            loss.backward()
            # Log to wandb
            wandb.log({"Training (mse)": loss})
            optimizer.step()
            print("Training mse = ",loss.item())
            train_loss+=loss.item()

        print("Validation loop - epoch number ",e)
        with torch.no_grad():
            valid_loss = 0.0
            val_set_size = int(val_percent*image_reader.count)
            start = train_set_size
            end=start+val_set_size
            for emb_images,emb_text in zip(image_reader(batch_size=batch_size, start=start, end=end),
                    text_reader(batch_size=batch_size, start=start, end=end)):
                emb_images = list(emb_images)
                emb_text = list(emb_text)
                emb_images[0] = torch.tensor(emb_images[0]).to(device)
                emb_text[0] = torch.tensor(emb_text[0]).to(device)
                diffusion_prior.eval()
                loss = diffusion_prior(text_embed = emb_text[0],image_embed = emb_images[0])

                # Log to wandb
                wandb.log({"Validation Loss(mse) ": loss})
                valid_loss+=loss.item()

        # Saving State Dict
        torch.save(diffusion_prior.state_dict(), 'saved_model.pth')

    test_set_size = int(test_percent*train_set_size)
    with torch.no_grad():
        for emb_images,emb_text in zip(image_reader(batch_size=batch_size, start=train_set_size+val_set_size,
            end=text_reader.count),text_reader(batch_size=batch_size, start=train_set_size+val_set_size, end=text_reader.count)):
            emb_images = list(emb_images)
            emb_text = list(emb_text)
            emb_images[0] = torch.tensor(emb_images[0]).to(device)
            emb_text[0] = torch.tensor(emb_text[0]).to(device)
            diffusion_prior.eval()
            loss = diffusion_prior(text_embed = emb_text[0],image_embed = emb_images[0])
            # Log to wandb
            wandb.log({"Test mse ": loss})

def main():
    parser = argparse.ArgumentParser()
    # Logging
    parser.add_argument("--wandb-entity", type=str, default="laion")
    parser.add_argument("--wandb-project", type=str, default="diffusion-prior")
    parser.add_argument("--wandb-name", type=str, default="laion-dprior")
    parser.add_argument("--wandb-dataset", type=str, default="LAION-5B")


    # URLs for embeddings 
    parser.add_argument("--image-embed-url", type=str, default="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/img_emb/")
    parser.add_argument("--text-embed-url", type=str, default="https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/text_emb/ ")
    # Hyperparameters
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=10**4)
    parser.add_argument("--loss-type", type=str, default="l2")
    parser.add_argument("--num-epochs", type=int, default=5)
    # Image embed dimension
    parser.add_argument("--image-embed-dim", type=int, default=768)
    # Train-test split
    parser.add_argument("--train-percent", type=float, default=0.7)
    parser.add_argument("--val-percent", type=float, default=0.2)
    parser.add_argument("--test-percent", type=float, default=0.1)
    # LAION training(pre-computed embeddings)
    parser.add_argument("--condition-on-text-encodings", type=bool, default=False)

    args = parser.parse_args()
    print("Setting up wandb logging... Please wait...")
    wandb.init(
      entity=args.wandb_entity,
      project=args.wandb_project,
      name=args.wandb_name,
      config={
      "learning_rate": args.learning_rate,
      "architecture": "DiffusionPrior",
      "dataset": args.wandb_dataset,
      "epochs": 5,
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
          args.loss_type,
          args.condition_on_text_encodings,
          device,args.learning_rate)

if __name__ == "__main__":
  main()
