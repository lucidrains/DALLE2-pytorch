import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse('2.0.0'):
    from einops._torch_specific import allow_ops_in_compiled_graph
    allow_ops_in_compiled_graph()

from dalle2_pytorch.version import __version__
from dalle2_pytorch.dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder
from dalle2_pytorch.dalle2_pytorch import OpenAIClipAdapter, OpenClipAdapter
from dalle2_pytorch.trainer import DecoderTrainer, DiffusionPriorTrainer

from dalle2_pytorch.vqgan_vae import VQGanVAE
from x_clip import CLIP
