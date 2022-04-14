import torch
import torch.utils.data
from torch import nn

from dalle2_pytorch import CLIP


class CLIPTrainer:
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
        log_interval=10,
    ):
        super().__init__()
        self.log_interval = log_interval
        self.freeze_image_encoder = freeze_image_encoder
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

    def shared_step(self, texts, images):
        loss = self.clip(
            texts,
            images,
            freeze_image_encoder=self.freeze_image_encoder,
            return_loss=True,
        )
        return loss

    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def train_epoch(self, epoch: int, train_dataloader):
        self.clip.train()
        for batch_idx, (texts, images) in enumerate(train_dataloader):
            texts, images = texts.to(self.device), images.to(self.device)
            loss = self.shared_step(texts, images)
            if (batch_idx == 0) or ((batch_idx + 1) % self.log_interval == 0):
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(images),
                        len(train_dataloader.dataset),
                        100.0 * batch_idx / len(train_dataloader),
                        loss.item(),
                    )
                )

    def val_epoch(self, epoch: int, val_dataloader):
        if val_dataloader is None:
            return
        self.clip.eval()
        test_loss = 0
        with torch.no_grad():
            for texts, images in val_dataloader:
                texts, images = texts.to(self.device), images.to(self.device)
                test_loss = self.shared_step(texts, images)

        test_loss /= len(val_dataloader.dataset)

        print(
            "\nTest Epoch: Average loss: {:.4f}\n".format(
                epoch, test_loss, len(val_dataloader.dataset)
            )
        )

    def run(self, max_epochs: int, train_dataloader, val_dataloader=None):
        for epoch in range(1, max_epochs + 1):
            self.train_epoch(epoch, train_dataloader=train_dataloader)
            self.val_epoch(epoch, val_dataloader=val_dataloader)


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
    clip_trainer = CLIPTrainer()
    clip_trainer.run(1, dataloader)
