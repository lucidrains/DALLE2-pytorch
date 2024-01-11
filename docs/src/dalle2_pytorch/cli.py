import click
import torch
import torchvision.transforms as T
from functools import reduce
from pathlib import Path

from dalle2_pytorch import DALLE2, Decoder, DiffusionPrior

def safeget(dictionary, keys, default = None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split('.'), dictionary)

def simple_slugify(text, max_length = 255):
    return text.replace("-", "_").replace(",", "").replace(" ", "_").replace("|", "--").strip('-_')[:max_length]

def get_pkg_version():
    from pkg_resources import get_distribution
    return get_distribution('dalle2_pytorch').version

def main():
    pass

@click.command()
@click.option('--model', default = './dalle2.pt', help = 'path to trained DALL-E2 model')
@click.option('--cond_scale', default = 2, help = 'conditioning scale (classifier free guidance) in decoder')
@click.argument('text')
def dream(
    model,
    cond_scale,
    text
):
    model_path = Path(model)
    full_model_path = str(model_path.resolve())
    assert model_path.exists(), f'model not found at {full_model_path}'
    loaded = torch.load(str(model_path))

    version = safeget(loaded, 'version')
    print(f'loading DALL-E2 from {full_model_path}, saved at version {version} - current package version is {get_pkg_version()}')

    prior_init_params = safeget(loaded, 'init_params.prior')
    decoder_init_params = safeget(loaded, 'init_params.decoder')
    model_params = safeget(loaded, 'model_params')

    prior = DiffusionPrior(**prior_init_params)
    decoder = Decoder(**decoder_init_params)

    dalle2 = DALLE2(prior, decoder)
    dalle2.load_state_dict(model_params)

    image = dalle2(text, cond_scale = cond_scale)

    pil_image = T.ToPILImage()(image)
    return pil_image.save(f'./{simple_slugify(text)}.png')
