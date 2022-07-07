# Diffusion Prior
This readme serves as an introduction to the diffusion prior.

## Intro

A properly trained prior will allow you to translate between two embedding spaces. If you know *a priori* that two embeddings are connected some way—then ability the translate between them could extremely helpful.

### Motivation

Before we dive into the model, let’s look at a quick example of where the model may be helpful.

For demonstration purposes we will imagine that we wish to generate images from text using CLIP and a Decoder.

> [CLIP](https://openai.com/blog/clip/) is a contrastive model that learns to maximize the cosine similarity between a given image and caption, however, there is no guarantee that these embeddings are in the same space. While the embeddings generated are ***close*** the image and text embeddings occupy two disjoint sets.

```python
# Load Models
clip_model = clip.load("ViT-L/14")
decoder = Decoder(checkpoint="best.pth") # A decoder trained on CLIP Image embeddings

# Retrieve prompt from user and encode with CLIP
prompt = "A corgi wearing sunglasses"
tokenized_text = tokenize(prompt)
text_embedding = clip_model.encode_text(tokenized_text)

# Now, pass the text embedding to the decoder
predicted_image = decoder.sample(text_embedding)
```

> **Question**: *Can you spot the issue here?*
>
> **Answer**: *We’re trying to generate an image from a text embedding!*

Unfortunately, we run into the issue previously mentioned--the image embeddings and the text embeddings are not interchangeable! Now let's look at a better solution

```python
# Load Models
prior= Prior(checkpoint="prior.pth") # A decoder trained to go from: text-> clip text emb -> clip img emb
decoder = Decoder(checkpoint="decoder.pth") # A decoder trained on CLIP Image embeddings

# Retrieve prompt from user and encode with a prior
prompt = "A corgi wearing sunglasses"
tokenized_text = tokenize(prompt)
text_embedding = prior.sample(tokenized_text) # <-- now we get an embedding in the same space as images!

# Now, pass the predicted image embedding to the decoder
predicted_image = decoder.sample(text_embedding)
```

With the prior we are able to successfully generate embeddings *within* CLIP's image space! For this reason, the decoder will perform much better as it receives input that is much closer to its training data.

> **You may be asking yourself the following question:**
>
> *"Why don't you just train the decoder on clip text embeddings instead of image embeddings?"*
>
> OpenAI covers this topic in their [DALLE-2 paper](https://arxiv.org/abs/2204.06125). The TL;DR is *"it doesn't work as well as decoders trained on image embeddings"*...also...its just an example :smile:

## Usage

To utilize a pre-trained prior, it’s quite simple.

### Loading Checkpoints
```python
import torch
from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork, OpenAIClipAdapter
from dalle2_pytorch.trainer import DiffusionPriorTrainer

def load_diffusion_model(dprior_path):

    prior_network = DiffusionPriorNetwork(
        dim=768,
        depth=24,
        dim_head=64,
        heads=32,
        normformer=True,
        attn_dropout=5e-2,
        ff_dropout=5e-2,
        num_time_embeds=1,
        num_image_embeds=1,
        num_text_embeds=1,
        num_timesteps=1000,
        ff_mult=4
    )

    diffusion_prior = DiffusionPrior(
        net=prior_network,
        clip=OpenAIClipAdapter("ViT-L/14"),
        image_embed_dim=768,
        timesteps=1000,
        cond_drop_prob=0.1,
        loss_type="l2",
        condition_on_text_encodings=True,

    )

    trainer = DiffusionPriorTrainer(
        diffusion_prior=diffusion_prior,
        lr=1.1e-4,
        wd=6.02e-2,
        max_grad_norm=0.5,
        amp=False,
        group_wd_params=True,
        use_ema=True,
        device=device,
        accelerator=None,
    )

    trainer.load(dprior_path)

    return trainer
```

 Here we instantiate a model matches the configuration it was trained with, and then load the weights (*just like any other PyTorch model!*)

### Sampling
Once we have a pre-trained model, generating embeddings is quite simple!
```python
# tokenize the text
tokenized_text = clip.tokenize("<your amazing prompt>")
# predict an embedding
predicted_embedding = prior.sample(tokenized_text, n_samples_per_batch=2, cond_scale=1.0)
```

The resulting tensor returned from `.sample()` is of the same shape as your training data along the non-batch dimension(s). For example, a prior trained on `ViT-L/14` embeddings will predict an embedding of shape (1, 768).

> For CLIP priors, this is quite handy as it means that you can use prior.sample(tokenizer_text) as a drop in replacement for clip.encode_text().

**Some things to note:**
* It is possible to specify the number of embeddings to sample from (the default suggested by OpenAI is `n=2`). Put simply, the idea here is that you avoid getting unlucky with a bad embedding generation by creating two; and selecting the one with the higher cosine similarity with the prompt.
* You may specify a higher conditioning scale than the default (`1.0`). It is unclear whether OpenAI uses a higher value for the prior specifically, or only on the decoder. Local testing has shown poor results with anything higher than `1.0` but *ymmv*.

---

## Training

### Overview

Training the prior is a relatively straightforward process thanks to the Trainer base class. The major step that is required of you is preparing a dataset in the format that EmbeddingReader expects. Having pre-computed embeddings massively increases training efficiency and is generally recommended as you will likely benefit from having them on hand for other tasks as well. Once you have a dataset, you are ready to move onto configuration

## Dataset

To train the prior, it is highly recommended to use precomputed embeddings for the images. To obtain these for a custom dataset, you can leverage [img2datset](https://github.com/rom1504/img2dataset) to pull images from a list of URLs and [clip_retrieval](https://github.com/rom1504/clip-retrieval#clip-inference) for generating the actual embeddings that can be used in the prior's dataloader.

## Configuration

The configuration file allows for you to easily track and reproduce experiments. It is a simple JSON file that will specify the architecture, dataset, and training parameters. For more information and specifics please see the configuration README.

## Distributed Training

If you would like to train in a distributed manner we have opted to leverage huggingface’ new Accelerate library. HFA makes it extremely simple to distribute work across multiple GPU’s and nodes. All that is required of you is to follow the simple CLI configuration tool [more information here](https://huggingface.co/docs/accelerate/accelerator).

## Evaluation

There are a variety of metrics available to you when training the prior. You can read a brief description of each in the table below:
| Metric                              | Description                                                                                                                                                                                                                                                  | Comments                                                                                                                                                                                                                                                                                                                                                |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Online Model Validation             | The validation loss associated with your online model.                                                                                                                                                                                                       | Ideally validation loss will be as low as possible. Using L2 loss, values as low as `0.1` and lower are possible after around 1 Billion samples seen.                                                                                                                                                                                                |
| EMA Validation                      | This metric measures the validation loss associated with your EMA model.                                                                                                                                                                                     | This will likely lag behind your "online" model's validation loss, but should outperform in the long-term.                                                                                                                                                                                                                                              |
| Baseline Similarity                 | Baseline similarity refers to the similarity between your dataset's prompts and associated image embeddings. This will serve as a guide for your prior's performance in cosine similarity.                                                                    | Generally `0.3` is considered a good cosine similarity for caption similarity.                                                                                                                                                                                                                                                                         |
| Similarity With Original Image      | This metric will measure the cosine similarity between your prior's predicted image embedding and the actual image that the caption was associated with. This is useful for determining wether your prior is generating images with the right contents.      | Values around `0.75`+ are obtainable. This metric should improve rapidly in the early stages of training and plateau with diminishing increases over time. If it takes hundreds of millions of samples to reach above `0.5`/`0.6` similarity--then you likely are suffering from some kind of training error or inefficiency (i.e. not using EMA) |
| Difference From Baseline Similarity | Sometimes its useful to visualize a metric in another light. This metric will show you how your prior's predicted image embeddings match up with the baseline similarity measured in your dataset.                                                           | This value should float around `0.0` with some room for variation. After a billion samples seen, values are within `0.01`+/- of `0.0`. If this climbs to high, (~>`0.02`) then this may be a sign that your model is overfitting somehow.                                                                                                       |
| Similarity With Text                | This metric is your bread and butter cosine similarity between the predicted image embedding and the original caption given to the prior. Monitoring this metric will be on of your main focuses and is probably the second most important behind your loss. | As mentioned, this value should be close to baseline similarity. We have observed early rapid increase with diminishing returns as the prior learns to generate valid image embeddings. If this value increases too far beyond the baseline similarity--it could be an indication that your model is overfitting.                                       |
| Similarity With Unrelated Caption   | This metric will attempt to exposed an overfit prior by feeding it arbitrary prompts (from your dataset) and then measure the similarity of this predicted embedding with some other image.                                                                   | Early on we found that a poorly trained/modeled prior could effectively fool CLIP into believing that the cosine similarity between two images were high (when in fact the caption and image were completely unrelated). With this in mind--a low value is ideal, anything below `0.1` is probably safe.                                              |

## Launching the script

Now that you’ve done all the prep it’s time for the easy part! 🚀

To actually launch the script, you will either use `accelerate launch train_diffusion_prior.py --config_path <path to your config>` to launch with distributed training & huggingface accelerate or `python train_diffusion_prior.py` if you would like to train on your gpu/cpu without huggingface accelerate.

## Checkpointing

Checkpoints will be saved to the directory specified in your configuration file.

Additionally, a final checkpoint is saved before running the test split. This file will be saved to the same directory and titled “latest.pth”. This is to avoid problems where your `save_every` configuration does not overlap with the number of steps required to do a complete pass through the data.

## Things To Keep In Mind

The prior has not been trained for tasks other than the traditional CLIP embedding translation…at least yet.

As we finalize the replication of unCLIP, there will almost assuredly be experiments attempting to apply the prior network to other tasks.

With that in mind, you are more or less a pioneer in embedding-translation if you are reading this and attempting something you don’t see documentation for!
