import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from crowdkit.aggregation import MajorityVote
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy
from tqdm.auto import tqdm

from annotator_model.utils import (
    get_mlp,
    batch_to_device,
    compute_worker_metrics,
    compute_worker_skills,
)
import plotly.express as px
from umap import UMAP
from pytorch_lightning import loggers as pl_loggers


class PlotEmbeddingsUMAP(Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        import mlflow

        if trainer.current_epoch < 1:
            return

        embeddings = (
            pl_module.annotator_embedding_transformer.worker_embedding.weight.cpu()
            .detach()
            .numpy()
        )
        reducer = UMAP(n_components=2)
        reduced = reducer.fit_transform(embeddings)
        fig = px.scatter(
            reduced,
            x=0,
            y=1,
            title=f"Worker embeddings UMAP, epoch {trainer.current_epoch}",
        )
        mlflow.log_figure(fig, f"umap_epoch_{trainer.current_epoch}.html")


class RecomputeWorkerSkills(Callback):
    def __init__(self, worker_skill_method, mlflow, start_recompute_skills_after_epoch):
        super().__init__()
        self.skills_history = {}
        self.worker_skill_method = worker_skill_method
        self.start_recompute_skills_after_epoch = start_recompute_skills_after_epoch
        self.mlflow = mlflow

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch < self.start_recompute_skills_after_epoch:
            return
        pl_module.eval()

        eval_workers = list(
            pl_module.worker_skills.sort_values(ascending=False)
            .head(int(len(pl_module.worker_skills) * 1 / 3))
            .index
        )
        preds = []
        labels_by_workers = []
        with torch.no_grad():
            loader = trainer.datamodule.train_dataloader(shuffle=False)[0]
            for batch_index, batch in enumerate(tqdm(loader, leave=False)):
                original_workers = batch["workers"].clone()
                original_labels = batch["labels"].clone()

                batch["workers"] = torch.LongTensor(eval_workers).repeat(
                    len(batch["workers"]), 1
                )
                batch = batch_to_device(batch, pl_module.device)
                sequences = pl_module.forward_training(
                    batch["images"], batch["workers"]
                )
                worker_logits = sequences[
                    :, 1:
                ]  # (batch_size, len(eval_workers), n_classes)
                worker_labels = worker_logits.argmax(-1).cpu().tolist()
                for task_index, worker_labels_list in enumerate(worker_labels):
                    task = batch_index * trainer.datamodule.batch_size + task_index
                    for i, l in enumerate(worker_labels_list):
                        preds.append(
                            {
                                "task": task,
                                "worker": eval_workers[i],
                                "label": l,
                            }
                        )

                    for worker, label in zip(
                        original_workers[task_index], original_labels[task_index]
                    ):
                        labels_by_workers.append(
                            {
                                "task": task,
                                "worker": worker.item(),
                                "label": label.item(),
                            }
                        )
        preds = pd.DataFrame(preds)
        mv_labels = MajorityVote().fit_predict(preds).to_dict()

        labels_by_workers_df = pd.DataFrame(labels_by_workers)
        labels_by_workers_df["true_label"] = labels_by_workers_df.task.apply(
            lambda t: mv_labels[t]
        )

        skills = compute_worker_skills(
            labels_by_workers_df, self.worker_skill_method, true_labels_included=True
        )

        self.skills_history[trainer.current_epoch] = pl_module.worker_skills.copy()
        pl_module.worker_skills = skills

        if self.mlflow:
            import mlflow

            fig = px.histogram(skills, title="Worker skills distribution")
            mlflow.log_figure(fig, f"worker_skills_hist_{trainer.current_epoch}.html")


class AnnotatorEmbeddingTransformer(nn.Module):
    def __init__(
        self,
        n_workers,
        worker_embedding_dim,
        task_dim,
        n_heads=4,
        n_encoder_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.n_workers = n_workers
        self.worker_embedding_dim = worker_embedding_dim
        self.task_dim = task_dim
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.dropout = dropout

        self.worker_embedding = nn.Embedding(n_workers, worker_embedding_dim)

        self.task_projector = nn.Linear(self.task_dim, self.worker_embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.worker_embedding_dim,
            nhead=self.n_heads,
            batch_first=True,
            dropout=dropout,
            dim_feedforward=self.worker_embedding_dim,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_encoder_layers,
            norm=nn.LayerNorm(normalized_shape=self.worker_embedding_dim),
        )

    def forward(self, task_features, worker_ids):
        task_features = F.normalize(
            self.task_projector(task_features.unsqueeze(1)), dim=-1
        )
        worker_embeddings = F.normalize(self.worker_embedding(worker_ids), dim=-1)
        sequence = torch.cat([task_features, worker_embeddings], dim=1)
        sequence = sequence * np.sqrt(self.worker_embedding_dim)
        sequence_encoded = self.encoder(sequence)
        return sequence_encoded


class AnnotatorEmbeddings(pl.LightningModule):
    def __init__(
        self,
        annotator_embedding_transformer=None,
        classifier=None,
        backbone_class=None,
        backbone_hparams=None,
        backbone_state_dict=None,
        num_labels=None,
        n_workers=None,
        worker_embedding_dim=20,
        worker_skills=None,
        top_k=7,
        forced_top_workers=None,
        lr=1e-5,
        n_epoch=10,
        n_epoch_batches=60000 // 128,
        dropout=0.1,
        label_smoothing=0.2,
        fixmatch_coef=1,
        fixmatch_softmax_t=1,
        fixmatch_threshold=0.95,
    ):
        super().__init__()
        self.n_workers = n_workers
        self.num_labels = num_labels
        self.worker_skills = worker_skills
        self.top_k = top_k
        self.forced_top_workers = forced_top_workers
        self.lr = lr
        self.n_epoch = n_epoch
        self.n_epoch_batches = n_epoch_batches
        self.num_labels = num_labels
        self.dropout = dropout
        self.worker_embedding_dim = worker_embedding_dim
        self.label_smoothing = label_smoothing
        self.fixmatch_coef = fixmatch_coef
        self.fixmatch_softmax_t = fixmatch_softmax_t
        self.fixmatch_threshold = fixmatch_threshold

        # Is this ugly? Yes
        # Is this necessary? Also yes
        # This is needed to be able to save an approach checkpoint together with the backbone
        # In lightning, you can only do this if the backbone is created in this constructor
        # If you just pass the backbone to the constructor, lightning will try to save the whole backbone state dict to hparams
        # Unfortunately this is the only way I found. I am sorry :(
        # Hack source: https://github.com/PyTorchLightning/pytorch-lightning/issues/7447#issuecomment-835695726
        self.backbone_class = backbone_class
        self.backbone_hparams = backbone_hparams or {}
        backbone_kwargs = dict(self.backbone_hparams)
        self.backbone_model = self.backbone_class(**backbone_kwargs)
        if backbone_state_dict:
            self.backbone_model.load_state_dict(backbone_state_dict)

        self.annotator_embedding_transformer = (
            annotator_embedding_transformer
            or AnnotatorEmbeddingTransformer(
                n_workers=self.n_workers,
                worker_embedding_dim=self.worker_embedding_dim,
                task_dim=list(self.backbone_model.parameters())[-1].shape[0],
                dropout=self.dropout,
            )
        )

        self.classifier = classifier or get_mlp(
            output_dim=self.num_labels,
            dropout_rate=self.dropout,
        )

        self.loss = nn.CrossEntropyLoss(
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        self.save_hyperparameters(
            ignore=["backbone_state_dict"]
        )  # saves to self.hparams

    def get_top_workers(self):
        if self.forced_top_workers:
            return self.forced_top_workers
        return list(
            self.worker_skills.sort_values(ascending=False).head(self.top_k).index
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            steps_per_epoch=self.n_epoch_batches,
            epochs=self.n_epoch,
            pct_start=0.4,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def configure_trainer(
        self,
        trainer,
        mlflow=False,
        mlflow_tracking_uri=None,
        wandb=False,
        start_recompute_skills_after_epoch=20,
        recompute_worker_skills=None,
        worker_skill_method="mv_prior",
    ):

        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            StochasticWeightAveraging(),
        ]

        if mlflow:
            callbacks.append(PlotEmbeddingsUMAP())
            logger_mlflow = pl_loggers.MLFlowLogger(
                tracking_uri=mlflow_tracking_uri,
            )
            trainer.loggers = list(trainer.loggers) + [logger_mlflow]

        if wandb:
            trainer.loggers = list(trainer.loggers) + [pl_loggers.WandbLogger()]

        skills_callback = None
        if recompute_worker_skills:
            skills_callback = RecomputeWorkerSkills(
                mlflow=mlflow,
                start_recompute_skills_after_epoch=start_recompute_skills_after_epoch,
                worker_skill_method=worker_skill_method,
            )
            callbacks.append(skills_callback)

        for callback in callbacks:
            trainer.callbacks.append(callback)

        return callbacks, trainer, skills_callback

    def _process_batch(self, batch, crowd_input=True):
        labels = batch["labels"]
        if crowd_input:
            sequence = self.forward_training(batch["images"], batch["workers"])
            logits = sequence[:, 1:]
        else:
            sequence = self.forward_inference(batch["images"])
            logits = sequence

        loss_classifier = self.loss(
            logits.reshape(-1, self.num_labels), labels.reshape(-1)
        )
        if "workers" in batch:
            workers = batch["workers"].reshape(-1)
            batch_skills = self.worker_skills.loc[workers.cpu().tolist()].values
            skills = torch.tensor(
                batch_skills,
                device=self.device,
            )
            skills = skills / skills.max()
            loss_classifier = loss_classifier * skills
        loss_classifier = loss_classifier.mean()

        return sequence, loss_classifier

    def forward_fixmatch(self, images_weak, images_strong, workers):
        logits_u_w = self.forward_training(images_weak, workers)[:, 1:]
        logits_u_s = self.forward_training(images_strong, workers)[:, 1:]
        pseudo_label = torch.softmax(
            logits_u_w.detach() / self.fixmatch_softmax_t, dim=-1
        )
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.fixmatch_threshold).float()
        logits = logits_u_s.reshape(-1, self.num_labels)
        targets = targets_u.reshape(-1)
        loss_fixmatch = (
            F.cross_entropy(
                logits,
                target=targets,
                reduction="none",
            )
            * mask.reshape(-1)
        ).mean()

        return logits_u_w, logits_u_s, mask, loss_fixmatch

    def forward_inference(self, images):
        backbone_output = self.backbone_model(images)
        batch_size = backbone_output.shape[0]

        top_worker_ids = self.get_top_workers()
        worker_ids = list(top_worker_ids)
        worker_ids = torch.tensor(worker_ids).to(backbone_output.device)
        worker_ids = worker_ids.unsqueeze(0).repeat_interleave(batch_size, 0)

        classifier_input = self.annotator_embedding_transformer(
            backbone_output, worker_ids
        )
        logits = self.classifier(classifier_input)

        worker_logits = logits[:, 1:, :]
        batch_size = worker_logits.shape[0]
        worker_logits = worker_logits.reshape(-1, self.num_labels)
        argmx = worker_logits.argmax(dim=-1)
        new_logits = torch.zeros(
            (worker_logits.shape[0], worker_logits.shape[-1]), device=argmx.device
        )
        new_logits = new_logits.scatter(1, argmx.unsqueeze(1), 1.0)
        new_logits = new_logits.reshape(batch_size, len(top_worker_ids), -1)

        # Scale predicted labels by skills
        # worker_skills = torch.tensor(
        #     self.worker_skills[top_worker_ids].values, device=new_logits.device
        # )
        # new_logits = new_logits * worker_skills.reshape(-1, 1)

        return new_logits.sum(dim=1)

    def forward_training(self, images, workers=None):
        hidden = self.backbone_model(images)
        sequence = self.annotator_embedding_transformer(hidden, workers)
        logits = self.classifier(sequence)
        return logits

    def forward(self, images, workers=None, **kwargs):
        if self.training:
            return self.forward_training(images, workers=workers)
        else:
            return self.forward_inference(images)

    def training_step(self, batch, batch_idx):
        labelled_batch = batch[0]
        fixmatch_batch = batch[1]
        logits, loss_classifier = self._process_batch(labelled_batch, crowd_input=True)
        self.log(
            "train_loss_classifier",
            loss_classifier,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        logits_weak, logits_strong, mask, loss_fixmatch = self.forward_fixmatch(
            fixmatch_batch["images_weak"],
            fixmatch_batch["images_strong"],
            fixmatch_batch["workers"],
        )
        loss_fixmatch = self.fixmatch_coef * loss_fixmatch
        self.log(
            "train_fixmatch_loss",
            loss_fixmatch,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.log(
            "fixmatch_mask_mean",
            mask.mean().item(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        train_loss = loss_classifier + loss_fixmatch
        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        scheduler_lr = self.trainer.lr_scheduler_configs[
            0
        ].scheduler.optimizer.param_groups[0]["lr"]
        self.log("lr", scheduler_lr, on_step=True, on_epoch=True, logger=True)

        return dict(
            loss=train_loss,
            loss_classifier=loss_classifier,
            loss_fixmatch=loss_fixmatch,
        )

    def validation_step(self, batch, batch_idx, dataloader_idx):
        split_name = ["val", "test"][dataloader_idx]
        if split_name == "test":
            return self.test_step(batch, batch_idx)

        logits, loss = self._process_batch(batch, crowd_input=True)
        acc = accuracy(
            logits[:, 1:].reshape(-1, self.num_labels), batch["labels"].flatten()
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log(
            "hp_metric", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        self.log(
            "val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "mean_skill",
            self.worker_skills.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        logits, loss = self._process_batch(batch, crowd_input=False)
        acc = accuracy(logits, batch["labels"])
        self.log(
            "test_acc", acc, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        logits, _ = self._process_batch(batch, crowd_input=False)
        predictions = torch.argmax(logits, dim=-1)
        return predictions, logits
