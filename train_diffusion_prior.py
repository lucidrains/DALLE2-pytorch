# TODO: add start, num_data_points, eval_every and group to config
# TODO: switch back to repo's wandb

START = 0
NUM_DATA_POINTS = 250e6
EVAL_EVERY = 1000
GROUP = "distributed"

import os
import click
import wandb

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np

from accelerate import Accelerator

from dalle2_pytorch.dataloaders import get_reader, make_splits
from dalle2_pytorch.utils import Timer
from dalle2_pytorch.train_configs import (
    DiffusionPriorTrainConfig,
    TrainDiffusionPriorConfig,
)
from dalle2_pytorch.trackers import BaseTracker, WandbTracker
from dalle2_pytorch import DiffusionPriorTrainer


# helpers


cos = nn.CosineSimilarity(dim=1, eps=1e-6)


def exists(val):
    return val is not None


def make_model(
    prior_config, train_config, device: str = None, accelerator: Accelerator = None
):
    # create model from config
    diffusion_prior = prior_config.create()

    # instantiate the trainer
    trainer = DiffusionPriorTrainer(
        diffusion_prior=diffusion_prior,
        lr=train_config.lr,
        wd=train_config.wd,
        max_grad_norm=train_config.max_grad_norm,
        amp=train_config.amp,
        use_ema=train_config.use_ema,
        device=device,
        accelerator=accelerator,
    )

    return trainer


# eval functions


def eval_model(
    trainer: DiffusionPriorTrainer,
    dataloader: DataLoader,
    text_conditioned: bool,
    loss_type: str,
    tracker_context: str,
    tracker: BaseTracker = None,
    use_ema: bool = True,
):
    trainer.eval()
    if trainer.is_main_process():
        click.secho(f"Measuring performance on {tracker_context}", fg="green", blink=True)

    with torch.no_grad():
        total_loss = 0.0
        total_samples = 0.0

        for image_embeddings, text_data in dataloader:
            image_embeddings = image_embeddings.to(trainer.device)
            text_data = text_data.to(trainer.device)

            batches = image_embeddings.shape[0]

            input_args = dict(image_embed=image_embeddings)

            if text_conditioned:
                input_args = dict(**input_args, text=text_data)
            else:
                input_args = dict(**input_args, text_embed=text_data)

            if use_ema:
                loss = trainer.ema_diffusion_prior(**input_args)
            else:
                loss = trainer(**input_args)

            total_loss += loss * batches
            total_samples += batches

        avg_loss = total_loss / total_samples

        stats = {f"{tracker_context}-{loss_type}": avg_loss}
        trainer.print(stats)

        if exists(tracker):
            tracker.log(stats, step=trainer.step.item() + 1)


def report_cosine_sims(
    trainer: DiffusionPriorTrainer,
    dataloader: DataLoader,
    text_conditioned: bool,
    tracker: BaseTracker,
    tracker_context: str = "validation",
):
    trainer.eval()
    if trainer.is_main_process():
        click.secho("Measuring Cosine-Similarity", fg="green", blink=True)

    for test_image_embeddings, text_data in dataloader:
        test_image_embeddings = test_image_embeddings.to(trainer.device)
        text_data = text_data.to(trainer.device)

        # we are text conditioned, we produce an embedding from the tokenized text
        if text_conditioned:
            text_embedding, text_encodings, text_mask = trainer.embed_text(text_data)
            text_cond = dict(
                text_embed=text_embedding, text_encodings=text_encodings, mask=text_mask
            )
        else:
            text_embedding = text_data
            text_cond = dict(text_embed=text_embedding)

        # make a copy of the text embeddings for shuffling
        text_embed_shuffled = text_embedding.clone()

        # roll the text to simulate "unrelated" captions
        rolled_idx = torch.roll(torch.arange(text_embedding.shape[0]), 1)
        text_embed_shuffled = text_embed_shuffled[rolled_idx]
        text_embed_shuffled = text_embed_shuffled / text_embed_shuffled.norm(
            dim=1, keepdim=True
        )

        if text_conditioned:
            text_encodings_shuffled = text_encodings[rolled_idx]
            text_mask_shuffled = text_mask[rolled_idx]
        else:
            text_encodings_shuffled = None
            text_mask_shuffled = None

        text_cond_shuffled = dict(
            text_embed=text_embed_shuffled,
            text_encodings=text_encodings_shuffled,
            mask=text_mask_shuffled,
        )

        # prepare the text embedding
        text_embed = text_embedding / text_embedding.norm(dim=1, keepdim=True)

        # prepare image embeddings
        test_image_embeddings = test_image_embeddings / test_image_embeddings.norm(
            dim=1, keepdim=True
        )

        # predict on the unshuffled text embeddings
        predicted_image_embeddings = trainer.p_sample_loop(
            test_image_embeddings.shape, text_cond
        )

        predicted_image_embeddings = (
            predicted_image_embeddings
            / predicted_image_embeddings.norm(dim=1, keepdim=True)
        )

        # predict on the shuffled embeddings
        predicted_unrelated_embeddings = trainer.p_sample_loop(
            test_image_embeddings.shape, text_cond_shuffled
        )

        predicted_unrelated_embeddings = (
            predicted_unrelated_embeddings
            / predicted_unrelated_embeddings.norm(dim=1, keepdim=True)
        )

        # calculate similarities
        original_similarity = cos(text_embed, test_image_embeddings).cpu().numpy()
        predicted_similarity = cos(text_embed, predicted_image_embeddings).cpu().numpy()
        unrelated_similarity = (
            cos(text_embed, predicted_unrelated_embeddings).cpu().numpy()
        )
        predicted_img_similarity = (
            cos(test_image_embeddings, predicted_image_embeddings).cpu().numpy()
        )

        stats = {
            f"{tracker_context}/baseline similarity": np.mean(original_similarity),
            f"{tracker_context}/similarity with text": np.mean(predicted_similarity),
            f"{tracker_context}/similarity with original image": np.mean(
                predicted_img_similarity
            ),
            f"{tracker_context}/similarity with unrelated caption": np.mean(unrelated_similarity),
            f"{tracker_context}/difference from baseline similarity": np.mean(
                predicted_similarity - original_similarity
            ),
        }

        for k, v in stats.items():
            trainer.print(f"{tracker_context}/{k}: {v}")

        if exists(tracker):
            tracker.log(stats, step=trainer.step.item() + 1)


# training script


def train(
    trainer: DiffusionPriorTrainer,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    test_loader: DataLoader,
    config: DiffusionPriorTrainConfig,
):
    # distributed tracking with wandb
    if trainer.accelerator.num_processes > 1:
        os.environ["WANDB_START_METHOD"] = "thread"

    tracker = wandb.init(
        name=f"RANK:{trainer.device}",
        entity=config.tracker.wandb_entity,
        project=config.tracker.wandb_project,
        config=config.dict(),
        group=GROUP,
    )

    # sync after tracker init
    trainer.wait_for_everyone()

    # init a timer
    timer = Timer()

    # do training
    for img, txt in train_loader:
        trainer.train()
        current_step = trainer.step.item() + 1

        # place data on device
        img = img.to(trainer.device)
        txt = txt.to(trainer.device)

        # pass to model
        loss = trainer(text=txt, image_embed=img)

        # display & log loss (will only print from main process)
        trainer.print(f"Step {current_step}: Loss {loss}")

        # perform backprop & apply EMA updates
        trainer.update()

        # track samples/sec/rank
        samples_per_sec = img.shape[0] / timer.elapsed()

        # samples seen
        samples_seen = (
            config.data.batch_size * trainer.accelerator.num_processes * current_step
        )

        # ema decay
        ema_decay = trainer.ema_diffusion_prior.get_current_decay()

        # Log on all processes for debugging
        tracker.log(
            {
                "tracking/samples-sec": samples_per_sec,
                "tracking/samples-seen": samples_seen,
                "tracking/ema-decay": ema_decay,
                "metrics/training-loss": loss,
            },
            step=current_step,
        )

        # Metric Tracking & Checkpointing (outside of timer's scope)
        if current_step % EVAL_EVERY == 0:
            eval_model(
                trainer=trainer,
                dataloader=eval_loader,
                text_conditioned=config.prior.condition_on_text_encodings,
                loss_type=config.prior.loss_type,
                tracker_context="metrics/online-model-validation",
                tracker=tracker,
                use_ema=False,
            )

            eval_model(
                trainer=trainer,
                dataloader=eval_loader,
                text_conditioned=config.prior.condition_on_text_encodings,
                loss_type=config.prior.loss_type,
                tracker_context="metrics/ema-model-validation",
                tracker=tracker,
                use_ema=True,
            )

            report_cosine_sims(
                trainer=trainer,
                dataloader=eval_loader,
                text_conditioned=config.prior.condition_on_text_encodings,
                tracker=tracker,
                tracker_context="metrics",
            )

        if current_step % config.train.save_every == 0:
            trainer.save(f"{config.tracker.data_path}/chkpt_step_{current_step}.pth")

        # reset timer for next round
        timer.reset()

    # evaluate on test data

    eval_model(
        trainer=trainer,
        dataloader=test_loader,
        text_conditioned=config.prior.condition_on_text_encodings,
        loss_type=config.prior.loss_type,
        tracker_context="test",
        tracker=tracker,
    )

    report_cosine_sims(
        trainer,
        test_loader,
        config.prior.condition_on_text_encodings,
        tracker,
        tracker_context="test",
    )


def initialize_training(config, accelerator=None):
    """
    Parse the configuration file, and prepare everything necessary for training
    """

    # get a device

    if accelerator:
        device = accelerator.device
        click.secho(f"Accelerating on: {device}", fg="yellow")
    else:
        if torch.cuda.is_available():
            click.secho("GPU detected, defaulting to cuda:0", fg="yellow")
            device = "cuda:0"
        else:
            click.secho("No GPU detected...using cpu", fg="yellow")
            device = "cpu"

    # make the trainer (will automatically distribute if possible & configured)

    trainer = make_model(config.prior, config.train, device, accelerator).to(device)

    # reload from chcekpoint

    if config.load.resume == True:
        click.secho(f"Loading checkpoint: {config.load.source}", fg="cyan")
        trainer.load(config.load.source)

    # fetch and prepare data

    if trainer.is_main_process():
        click.secho("Grabbing data from source", fg="blue", blink=True)

    img_reader = get_reader(
        text_conditioned=trainer.text_conditioned,
        img_url=config.data.image_url,
        meta_url=config.data.meta_url,
    )

    train_loader, eval_loader, test_loader = make_splits(
        text_conditioned=trainer.text_conditioned,
        batch_size=config.data.batch_size,
        num_data_points=NUM_DATA_POINTS,
        train_split=config.data.splits.train,
        eval_split=config.data.splits.val,
        image_reader=img_reader,
        rank=accelerator.state.process_index if exists(accelerator) else 0,
        world_size=accelerator.state.num_processes if exists(accelerator) else 1,
        start=START,
    )

    # wait for everyone to load data before continuing
    trainer.wait_for_everyone()

    # start training
    train(
        trainer=trainer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        test_loader=test_loader,
        config=config,
    )


@click.command()
@click.option("--hfa", default=True)
@click.option("--config_path", default="configs/prior.json")
def main(hfa, config_path):
    # start HFA if requested
    if hfa:
        accelerator = Accelerator()
    else:
        accelerator = None

    # load the configuration file on main process
    if not exists(accelerator) or accelerator.is_main_process:
        click.secho(f"Loading configuration from {config_path}", fg="green")

    config = TrainDiffusionPriorConfig.from_json_path(config_path)

    # send config to get processed
    initialize_training(config, accelerator)


if __name__ == "__main__":
    main()
