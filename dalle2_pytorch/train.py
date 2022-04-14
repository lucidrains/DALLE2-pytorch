import torch
from dalle2_pytorch import CLIP

import pytorch_lightning as pl
import torch.utils.data
from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn
from x_clip import CLIP


class CLIPModule(pl.LightningModule):
    def __init__(
            self,
            image_encoder: nn.Module = None,
            text_encoder: nn.Module = None,
            dim_text=512,
            dim_image=512,
            dim_latent=512,
            num_text_tokens=10000,
            text_enc_depth=6,
            text_seq_len=256,
            text_heads=8,
            visual_enc_depth=6,
            visual_image_size=256,
            visual_patch_size=32,
            visual_heads=8,
            use_all_token_embeds=True,
            decoupled_contrastive_learning=True,
            extra_latent_projection=True,
            use_visual_ssl=True,
            visual_ssl_type="simclr",
            use_mlm=False,
            text_ssl_loss_weight=0.05,
            image_ssl_loss_weight=0.05,
            freeze_image_encoder: bool = False,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.freeze_image_encoder = freeze_image_encoder
        self.save_hyperparameters()
        self.clip = CLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            dim_text=dim_text,
            dim_image=dim_image,
            dim_latent=dim_latent,
            num_text_tokens=num_text_tokens,
            text_enc_depth=text_enc_depth,
            text_seq_len=text_seq_len,
            text_heads=text_heads,
            visual_enc_depth=visual_enc_depth,
            visual_image_size=visual_image_size,
            visual_patch_size=visual_patch_size,
            visual_heads=visual_heads,
            use_all_token_embeds=use_all_token_embeds,
            decoupled_contrastive_learning=decoupled_contrastive_learning,
            extra_latent_projection=extra_latent_projection,
            use_visual_ssl=use_visual_ssl,
            visual_ssl_type=visual_ssl_type,
            use_mlm=use_mlm,
            text_ssl_loss_weight=text_ssl_loss_weight,
            image_ssl_loss_weight=image_ssl_loss_weight,
        )

    def forward(self, texts, images):
        loss = self.clip(
            texts,
            images,
            freeze_image_encoder=self.freeze_image_encoder,
            return_loss=True,
        )
        return loss

    def training_step(self, batch):
        texts, images = batch
        loss = self(texts, images)
        self.log("train/loss", loss)

    def validation_step(self, batch):
        texts, images = batch
        loss = self(texts, images)
        self.log("val/loss", loss)

    def configure_optimizers(self):
        """using manual optimization"""


def cli_main(
        datamodule=None, train_dataloader=None, val_dataloader=None, test_dataloader=None
):
    cli = LightningCLI(CLIPModule, save_config_overwrite=True, run=False)
    cli.trainer.fit(
        cli.model,
        datamodule=datamodule,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    if test_dataloader:
        cli.trainer.test(ckpt_path="best", test_dataloader=test_dataloader)


if __name__ == "__main__":
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self):
            super(MockDataset, self).__init__()

        def __getitem__(self, item):
            text = torch.randint(0, 10000, [256])
            images = torch.randn(3, 256, 256)
            return text, images

        def __len__(self):
            return 10


    dataloader = torch.utils.data.DataLoader(MockDataset(), batch_size=4)
    cli_main(train_dataloader=dataloader)
