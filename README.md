<img src="./dalle2.png" width="450px"></img>

## DALL-E 2 - Pytorch (wip)

Implementation of <a href="https://openai.com/dall-e-2/">DALL-E 2</a>, OpenAI's updated text-to-image synthesis neural network, in Pytorch.

<a href="https://youtu.be/RJwPN4qNi_Y?t=555">Yannic Kilcher summary</a> | <a href="https://www.youtube.com/watch?v=F1X4fHzF4mQ">AssemblyAI explainer</a>

The main novelty seems to be an extra layer of indirection with the prior network (whether it is an autoregressive transformer or a diffusion network), which predicts an image embedding based on the text embedding from CLIP. Specifically, this repository will only build out the diffusion prior network, as it is the best performing variant (but which incidentally involves a causal transformer as the denoising network 😂)

This model is SOTA for text-to-image for now.

It may also explore an extension of using <a href="https://huggingface.co/spaces/multimodalart/latentdiffusion">latent diffusion</a> in the decoder from Rombach et al.

Please join <a href="https://discord.gg/xBPBXfcFHd"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a> if you are interested in helping out with the replication

There was enough interest for a Jax version. It will be completed after the Pytorch version shows signs of life on my toy tasks. <a href="https://github.com/lucidrains/dalle2-jax">Placeholder repository</a>

## Install

```bash
$ pip install dalle2-pytorch
```

## Usage

To train DALLE-2 is a 3 step process, with the training of CLIP being the most important

To train CLIP, you can either use <a href="https://github.com/lucidrains/x-clip">x-clip</a> package, or join the LAION discord, where a lot of replication efforts are already <a href="https://github.com/mlfoundations/open_clip">underway</a>.

This repository will demonstrate integration with `x-clip` for starters

```python
import torch
from dalle2_pytorch import CLIP

clip = CLIP(
    dim_text = 512,
    dim_image = 512,
    dim_latent = 512,
    num_text_tokens = 49408,
    text_enc_depth = 1,
    text_seq_len = 256,
    text_heads = 8,
    visual_enc_depth = 1,
    visual_image_size = 256,
    visual_patch_size = 32,
    visual_heads = 8,
    use_all_token_embeds = True,            # whether to use fine-grained contrastive learning (FILIP)
    decoupled_contrastive_learning = True,  # use decoupled contrastive learning (DCL) objective function, removing positive pairs from the denominator of the InfoNCE loss (CLOOB + DCL)
    extra_latent_projection = True,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_visual_ssl = True,                  # whether to do self supervised learning on iages
    visual_ssl_type = 'simclr',             # can be either 'simclr' or 'simsiam', depending on using DeCLIP or SLIP
    use_mlm = False,                        # use masked language learning (MLM) on text (DeCLIP)
    text_ssl_loss_weight = 0.05,            # weight for text MLM loss
    image_ssl_loss_weight = 0.05            # weight for image self-supervised learning loss
).cuda()

# mock data

text = torch.randint(0, 49408, (4, 256)).cuda()
images = torch.randn(4, 3, 256, 256).cuda()

# train

loss = clip(
    text,
    images,
    return_loss = True              # needs to be set to True to return contrastive loss
)

loss.backward()

# do the above with as many texts and images as possible in a loop
```

Then, you will need to train the decoder, which learns to generate images based on the image embedding coming from the trained CLIP above

```python
import torch
from dalle2_pytorch import Unet, Decoder, CLIP

# trained clip from step 1

clip = CLIP(
    dim_text = 512,
    dim_image = 512,
    dim_latent = 512,
    num_text_tokens = 49408,
    text_enc_depth = 1,
    text_seq_len = 256,
    text_heads = 8,
    visual_enc_depth = 1,
    visual_image_size = 256,
    visual_patch_size = 32,
    visual_heads = 8
).cuda()

# unet for the decoder

unet = Unet(
    dim = 128,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults=(1, 2, 4, 8)
).cuda()

# decoder, which contains the unet and clip

decoder = Decoder(
    unet = unet,
    clip = clip,
    timesteps = 100,
    cond_drop_prob = 0.2
).cuda()

# mock images (get a lot of this)

images = torch.randn(4, 3, 256, 256).cuda()

# feed images into decoder

loss = decoder(images)
loss.backward()

# do the above for many many many many steps
# then it will learn to generate images based on the CLIP image embeddings
```

Finally, the main contribution of the paper. The repository offers the diffusion prior network. It takes the CLIP text embeddings and tries to generate the CLIP image embeddings. Again, you will need the trained CLIP from the first step

```python
import torch
from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior, CLIP

# get trained CLIP from step one

clip = CLIP(
    dim_text = 512,
    dim_image = 512,
    dim_latent = 512,
    num_text_tokens = 49408,
    text_enc_depth = 6,
    text_seq_len = 256,
    text_heads = 8,
    visual_enc_depth = 6,
    visual_image_size = 256,
    visual_patch_size = 32,
    visual_heads = 8,
).cuda()

# setup prior network, which contains an autoregressive transformer

prior_network = DiffusionPriorNetwork(
    dim = 512,
    depth = 6,
    dim_head = 64,
    heads = 8
).cuda()

# diffusion prior network, which contains the CLIP and network (with transformer) above

diffusion_prior = DiffusionPrior(
    net = prior_network,
    clip = clip,
    timesteps = 100,
    cond_drop_prob = 0.2
).cuda()

# mock data

text = torch.randint(0, 49408, (4, 256)).cuda()
images = torch.randn(4, 3, 256, 256).cuda()

# feed text and images into diffusion prior network

loss = diffusion_prior(text, images)
loss.backward()

# do the above for many many many steps
# now the diffusion prior can generate image embeddings from the text embeddings
```

In the paper, they actually used a <a href="https://cascaded-diffusion.github.io/">recently discovered technique</a>, from <a href="http://www.jonathanho.me/">Jonathan Ho</a> himself (original author of DDPMs, the core technique used in DALL-E v2) for high resolution image synthesis.

This can easily be used within this framework as so

```python
import torch
from dalle2_pytorch import Unet, Decoder, CLIP

# trained clip from step 1

clip = CLIP(
    dim_text = 512,
    dim_image = 512,
    dim_latent = 512,
    num_text_tokens = 49408,
    text_enc_depth = 6,
    text_seq_len = 256,
    text_heads = 8,
    visual_enc_depth = 6,
    visual_image_size = 256,
    visual_patch_size = 32,
    visual_heads = 8
).cuda()

# 2 unets for the decoder (a la cascading DDPM)

unet1 = Unet(
    dim = 32,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults = (1, 2, 4, 8)
).cuda()

unet2 = Unet(
    dim = 32,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults = (1, 2, 4, 8, 16)
).cuda()

# decoder, which contains the unet(s) and clip

decoder = Decoder(
    clip = clip,
    unet = (unet1, unet2),            # insert both unets in order of low resolution to highest resolution (you can have as many stages as you want here)
    image_sizes = (256, 512),         # resolutions, 256 for first unet, 512 for second. these must be unique and in ascending order (matches with the unets passed in)
    timesteps = 1000,
    cond_drop_prob = 0.2
).cuda()

# mock images (get a lot of this)

images = torch.randn(4, 3, 512, 512).cuda()

# feed images into decoder, specifying which unet you want to train
# each unet can be trained separately, which is one of the benefits of the cascading DDPM scheme

loss = decoder(images, unet_number = 1)
loss.backward()

loss = decoder(images, unet_number = 2)
loss.backward()

# do the above for many steps for both unets

# then it will learn to generate images based on the CLIP image embeddings

# chaining the unets from lowest resolution to highest resolution (thus cascading)

mock_image_embed = torch.randn(1, 512).cuda()
images = decoder.sample(mock_image_embed) # (1, 3, 512, 512)
```

Finally, to generate the DALL-E2 images from text. Insert the trained `DiffusionPrior` as well as the `Decoder` (which wraps `CLIP`, the causal transformer, and unet(s))

```python
from dalle2_pytorch import DALLE2

dalle2 = DALLE2(
    prior = diffusion_prior,
    decoder = decoder
)

# send the text as a string if you want to use the simple tokenizer from DALLE v1
# or you can do it as token ids, if you have your own tokenizer

texts = ['glistening morning dew on a flower petal']
images = dalle2(texts) # (1, 3, 256, 256)
```

That's it!

Let's see the whole script below

```python
import torch
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, CLIP

clip = CLIP(
    dim_text = 512,
    dim_image = 512,
    dim_latent = 512,
    num_text_tokens = 49408,
    text_enc_depth = 6,
    text_seq_len = 256,
    text_heads = 8,
    visual_enc_depth = 6,
    visual_image_size = 256,
    visual_patch_size = 32,
    visual_heads = 8
).cuda()

# mock data

text = torch.randint(0, 49408, (4, 256)).cuda()
images = torch.randn(4, 3, 256, 256).cuda()

# train

loss = clip(
    text,
    images,
    return_loss = True
)

loss.backward()

# do above for many steps ...

# prior networks (with transformer)

prior_network = DiffusionPriorNetwork(
    dim = 512,
    depth = 6,
    dim_head = 64,
    heads = 8
).cuda()

diffusion_prior = DiffusionPrior(
    net = prior_network,
    clip = clip,
    timesteps = 100,
    cond_drop_prob = 0.2
).cuda()

loss = diffusion_prior(text, images)
loss.backward()

# do above for many steps ...

# decoder (with unet)

unet1 = Unet(
    dim = 128,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults=(1, 2, 4, 8)
).cuda()

unet2 = Unet(
    dim = 16,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults = (1, 2, 4, 8, 16)
).cuda()

decoder = Decoder(
    unet = (unet1, unet2),
    image_sizes = (128, 256),
    clip = clip,
    timesteps = 100,
    cond_drop_prob = 0.2
).cuda()

for unet_number in (1, 2):
    loss = decoder(images, unet_number = unet_number) # this can optionally be decoder(images, text) if you wish to condition on the text encodings as well, though it was hinted in the paper it didn't do much
    loss.backward()

# do above for many steps

dalle2 = DALLE2(
    prior = diffusion_prior,
    decoder = decoder
)

images = dalle2(
    ['cute puppy chasing after a squirrel'],
    cond_scale = 2. # classifier free guidance strength (> 1 would strengthen the condition)
)

# save your image (in this example, of size 256x256)
```

Everything in this readme should run without error

You can also train the decoder on images of greater than the size (say 512x512) at which CLIP was trained (256x256). The images will be resized to CLIP image resolution for the image embeddings

For the layperson, no worries, training will all be automated into a CLI tool, at least for small scale training.

## CLI Usage (work in progress)

```bash
$ dream 'sharing a sunset at the summit of mount everest with my dog'
```

Once built, images will be saved to the same directory the command is invoked

## Training wrapper (wip)

Offer training wrappers

## Training CLI (wip)

<a href="https://github.com/lucidrains/stylegan2-pytorch">template</a>

## Todo

- [x] finish off gaussian diffusion class for latent embedding - allow for prediction of epsilon
- [x] add what was proposed in the paper, where DDPM objective for image latent embedding predicts x0 directly (reread vq-diffusion paper and get caught up on that line of work)
- [x] make sure it works end to end to produce an output tensor, taking a single gradient step
- [x] augment unet so that it can also be conditioned on text encodings (although in paper they hinted this didn't make much a difference)
- [x] figure out all the current bag of tricks needed to make DDPMs great (starting with the blur trick mentioned in paper)
- [x] build the cascading ddpm by having Decoder class manage multiple unets at different resolutions
- [ ] offload unets not being trained on to CPU for memory efficiency (for training each resolution unets separately)
- [ ] build out latent diffusion architecture in separate file, as it is not faithful to dalle-2 (but offer it as as setting)
- [ ] become an expert with unets, cleanup unet code, make it fully configurable, add efficient attention (conditional on resolution), port all learnings over to https://github.com/lucidrains/x-unet
- [ ] train on a toy task, offer in colab

## Citations

```bibtex
@misc{ramesh2022,
    title   = {Hierarchical Text-Conditional Image Generation with CLIP Latents}, 
    author  = {Aditya Ramesh et al},
    year    = {2022}
}
```

```bibtex
@misc{crowson2022,
    author  = {Katherine Crowson},
    url     = {https://twitter.com/rivershavewings}
}
```

```bibtex
@misc{rombach2021highresolution,
    title   = {High-Resolution Image Synthesis with Latent Diffusion Models}, 
    author  = {Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
    year    = {2021},
    eprint  = {2112.10752},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@inproceedings{Liu2022ACF,
    title   = {A ConvNet for the 2020s},
    author  = {Zhuang Liu and Hanzi Mao and Chaozheng Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
    year    = {2022}
}
```

```bibtex
@misc{zhang2019root,
    title   = {Root Mean Square Layer Normalization},
    author  = {Biao Zhang and Rico Sennrich},
    year    = {2019},
    eprint  = {1910.07467},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

*Creating noise from data is easy; creating data from noise is generative modeling.* - Yang Song's <a href="https://arxiv.org/abs/2011.13456">paper</a>
