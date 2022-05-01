import argparse
import os
from dalle2_pytorch import DiffusionPrior
from embedding_reader import EmbeddingReader
from dalle2_pytorch import DiffusionPriorNetwork
from dalle2_pytorch.optimizer import get_optimizer
import math
import time
from tqdm import tqdm
import torch
from torch import nn
import wandb
os.environ["WANDB_SILENT"] = "true"

def eval_model(model,device,image_reader,text_reader,start,end,batch_size,loss_type,phase="Validation"):
    with torch.no_grad():
        for emb_images,emb_text in zip(image_reader(batch_size=batch_size, start=start, end=end),
                text_reader(batch_size=batch_size, start=start, end=end)):
            emb_images_tensor = torch.tensor(emb_images[0]).to(device)
            emb_text_tensor = torch.tensor(emb_text[0]).to(device)
            model.eval()
            loss = model(text_embed = emb_text_tensor,image_embed = emb_images_tensor)

            # Log to wandb
            wandb.log({phase + " " + loss_type: loss})

def save_model(save_path,state_dict):
    # Saving State Dict
    print("====================================== Saving checkpoint ======================================")
    torch.save(state_dict, save_path+'/'+str(time.time())+'_saved_model.pth')

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
          save_interval,
          save_path,
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
    text_reader  = EmbeddingReader(embeddings_folder=text_embed_url, file_format="npy")
    num_data_points = text_reader.count

    # Create save_path if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ### Training code ###
    optimizer = get_optimizer(diffusion_prior.parameters())
    epochs = num_epochs

    step = 0
    t = time.time()
    train_set_size = int(train_percent*num_data_points)
    val_set_size = int(val_percent*num_data_points)
    for _ in range(epochs):
        for emb_images,emb_text in zip(image_reader(batch_size=batch_size, start=0, end=train_set_size),
                text_reader(batch_size=batch_size, start=0, end=train_set_size)):
            emb_images_tensor = torch.tensor(emb_images[0]).to(device)
            emb_text_tensor = torch.tensor(emb_text[0]).to(device)
            optimizer.zero_grad()
            loss = diffusion_prior(text_embed = emb_text_tensor,image_embed = emb_images_tensor)
            loss.backward()
            # Samples per second
            step+=1
            samples_per_sec = batch_size*step/(time.time()-t)
            # Save checkpoint every save_interval minutes
            if(int(time.time()-t) >= 60*save_interval):
                t = time.time()
                save_model(save_path,diffusion_prior.state_dict())
            # Log to wandb
            wandb.log({"Training loss": loss.item(),
                        "Steps": step,
                        "Samples per second": samples_per_sec})
            optimizer.step()

        ### Evaluate model(validation run) ###
        start = train_set_size
        end=start+val_set_size
        eval_model(diffusion_prior,device,image_reader,text_reader,start,end,batch_size,dp_loss_type,phase="Validation")

    ### Test run ###
    test_set_size = int(test_percent*train_set_size) 
    start=train_set_size+val_set_size
    end=num_data_points
    eval_model(diffusion_prior,device,image_reader,text_reader,start,end,batch_size,dp_loss_type,phase="Test")

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
    # Model checkpointing interval(minutes)
    parser.add_argument("--save-interval", type=int, default=30)
    parser.add_argument("--save-path", type=str, default="./diffusion_prior_checkpoints")

    args = parser.parse_args()
    print("Setting up wandb logging... Please wait...")
    wandb.init(
      entity=args.wandb_entity,
      project=args.wandb_project,
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
          args.save_interval,
          args.save_path,
          device,
          args.learning_rate)

if __name__ == "__main__":
  main()
