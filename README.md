<img src="./dalle2.png" width="450px"></img>

## DALL-E 2 - Pytorch (wip)

Implementation of <a href="https://openai.com/dall-e-2/">DALL-E 2</a>, OpenAI's updated text-to-image synthesis neural network, in Pytorch. <a href="https://youtu.be/RJwPN4qNi_Y?t=555">Yannic Kilcher summary</a>

The main novelty seems to be an extra layer of indirection with the prior network (whether it is an autoregressive transformer or a diffusion network), which predicts an image embedding based on the text embedding from CLIP. Specifically, this repository will only build out the diffusion prior network, as it is the best performing variant (but which incidentally involves a causal transformer as the denoising network 😂)

This model is SOTA for text-to-image for now.

It may also explore an extension of using <a href="https://huggingface.co/spaces/multimodalart/latentdiffusion">latent diffusion</a> in the decoder from Rombach et al.

Please join <a href="https://discord.gg/xBPBXfcFHd"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a> if you are interested in helping out with the replication

Do let me know if anyone is interested in a Jax version https://github.com/lucidrains/DALLE2-pytorch/discussions/8

For all of you emailing me (there is a lot), the best way to contribute is through pull requests. Everything is open sourced after all. All my thoughts are public. This is your moment to participate.

## Install

```bash
$ pip install dalle2-pytorch
```

## Usage (work in progress)

<a href="https://github.com/lucidrains/big-sleep">template</a>

```bash
$ dream 'sharing a sunset at the summit of mount everest with my dog'
```

Once built, images will be saved to the same directory the command is invoked

## Training (work in progress, will offer both in code and as command-line)

<a href="https://github.com/lucidrains/stylegan2-pytorch">template</a>

Todo

## Todo

- [x] finish off gaussian diffusion class for latent embedding - allow for prediction of epsilon
- [x] add what was proposed in the paper, where DDPM objective for image latent embedding predicts x0 directly (reread vq-diffusion paper and get caught up on that line of work)
- [ ] make sure it works end to end to produce an output tensor, taking a single gradient step
- [ ] augment unet so that it can also be conditioned on text encodings (although in paper they hinted this didn't make much a difference)
- [ ] look into Jonathan Ho's cascading DDPM for the decoder, as that seems to be what they are using. get caught up on DDPM literature
- [ ] figure out all the current bag of tricks needed to make DDPMs great (starting with the blur trick mentioned in paper)
- [ ] train on a toy task, offer in colab
- [ ] add attention to unet - apply some personal tricks with efficient attention

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
