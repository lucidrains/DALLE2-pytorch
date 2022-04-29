import torch
from PIL import Image

from dalle2_pytorch.dalle2_pytorch import BaseClipAdapter
import torchvision.transforms as T

def find_layer(model, layer):
    modules = dict([*model.named_modules()])
    return modules.get(layer, None)

def hook(_, input, output):
    print(output.shape)

import clip
# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).cuda()
image = torch.randn(1, 3, 224, 224).cuda()


class OpenAIClipAdapter(BaseClipAdapter):
    def __init__(self, name = 'ViT-B/32'):
        try:
            import clip
        except ImportError:
            print('you must install openai clip in order to use this adapter - `pip install git+https://github.com/openai/CLIP.git` - more instructions at https://github.com/openai/CLIP#usage')

        openai_clip, _ = clip.load(name)
        super().__init__(openai_clip)

        text_attention_final = self.find_layer(self.clip, 'ln_final')
        self.handle = text_attention_final.register_forward_hook(self._hook)
        self.clip_normalize = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.cleared = False

    def find_layer(self,  layer):
        modules = dict([*self.clip.named_modules()])
        return modules.get(layer, None)

    def clear(self):
        if self.cleared:
            return

        self.handle()

    def _hook(self, _, inputs, outputs):
        self.text_encodings = outputs

    @property
    def dim_latent(self):
        return 512

    @property
    def image_size(self):
        return self.clip.visual.input_resolution

    @property
    def image_channels(self):
        return 3

    @torch.no_grad()
    def embed_text(self, text):
        assert not self.cleared

        text_embed = self.clip.encode_text(text)
        text_encodings = self.text_encodings
        del self.text_encodings
        return text_embed, text_encodings

    @torch.no_grad()
    def embed_image(self, image):
        assert not self.cleared

        image = self.clip_normalize(image)
        image_embed = self.clip.encode_image(image)
        return image_embed, None

clip_adapter = OpenAIClipAdapter().cuda()

# print(model)
with torch.no_grad():
    image_features, _ = clip_adapter.embed_image(image)
    text_features, text_encodings = clip_adapter.embed_text(text)
    print(text_features.shape, image_features.shape)
    print(text_encodings.shape)
