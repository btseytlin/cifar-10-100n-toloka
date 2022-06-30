import argparse
import pickle
import time
from torchvision import transforms
import pytorch_lightning as pl
import torch
from torchvision.transforms import InterpolationMode

from annotator_model.backbone import ImageBackbone
from annotator_model.annotator_embeddings_model import AnnotatorEmbeddings
from annotator_model.datamodule import CIFAR10DataModule
from annotator_model.utils import (
    make_toloka_format_csvs,
    hparams_to_dict,
    cifar10_mean,
    cifar10_std,
)
from annotator_model.utils import set_global_seeds


def main():
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--noise_type",
        type=str,
        help="aggre, worst, rand1",
        choices=["aggre", "rand1", "worst"],
        default="aggre",
    )
    parser.add_argument(
        "--noise_path",
        type=str,
        help="path of CIFAR-10_human.pt",
        default="./data/CIFAR-10_human.pt",
    )
    parser.add_argument(
        "--side_info_path",
        type=str,
        default="./data/side_info_cifar10N.csv",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="only cifar10 is supported",
        default="cifar10",
        choices=["cifar10"],
    )
    parser.add_argument("--n_epoch", type=int, default=30)
    parser.add_argument(
        "--seed", type=int, default=0
    )  # we will test your code with 5 different seeds. The seeds are generated randomly and fixed for all participants.
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="how many subprocesses to use for data loading",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)

    # Lightning params
    parser.add_argument("--gpus", type=int, default=-1)
    parser.add_argument("--accelerator", type=str, default="auto")

    # Solution parameters
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.2)
    parser.add_argument("--fixmatch_coef", type=float, default=1)
    parser.add_argument("--fixmatch_threshold", type=float, default=0.85)
    parser.add_argument("--fixmatch_softmax_t", type=float, default=1)
    parser.add_argument("--debug_limit", type=int, default=None)
    parser.add_argument(
        "--worker_skill_method",
        type=str,
        default="mv_prior",
        choices=["zbs", "mv", "mv_prior"],
    )
    parser.add_argument(
        "--recompute_worker_skills",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="efficientnet_b1_pruned",
    )
    parser.add_argument("--start_recompute_skills_after_epoch", type=int, default=50)

    # ML flow
    parser.add_argument("--experiment_name", type=str, default="LMNL_AnnotatorModel")
    parser.add_argument("--mlflow", type=bool, default=False)
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--wandb_api_key", type=str, default=None)

    ##################################### main code ################################################
    args = parser.parse_args()

    # Seed
    set_global_seeds(args.seed)

    time_start = time.time()
    # Hyper Parameters
    learning_rate = args.lr
    noise_type_map = {
        "clean": "clean_label",
        "worst": "worse_label",
        "aggre": "aggre_label",
        "rand1": "random_label1",
        "rand2": "random_label2",
        "rand3": "random_label3",
        "clean100": "clean_label",
        "noisy100": "noisy_label",
    }
    args.noise_type = noise_type_map[args.noise_type]

    labels_csv_path, toloka_csv_path = make_toloka_format_csvs(
        side_info_df_path=args.side_info_path,
        noisy_labels_dict_path=args.noise_path,
        labels_df_output_path="data/labels.csv",
        toloka_df_output_path="data/train_toloka.csv",
    )

    input_size = 32
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=input_size,
                scale=(0.5, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandAugment(magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    datamodule = CIFAR10DataModule(
        root="./data/cifar10_data/",
        labels_csv_path=labels_csv_path,
        toloka_csv_path=toloka_csv_path,
        noise_type=args.noise_type,
        input_size=input_size,
        worker_skill_method=args.worker_skill_method,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True,
        debug_limit=args.debug_limit,
        train_transform=train_transform,
        val_transform=val_transform,
    )
    datamodule.setup()

    backbone_class = ImageBackbone
    backbone_kwargs = dict(model_name=args.backbone_name)

    backbone = backbone_class(**backbone_kwargs)

    dummy_loader = datamodule.train_dataloader()[0]

    model = AnnotatorEmbeddings(
        backbone_class=backbone_class,
        backbone_hparams=backbone.hparams,
        worker_skills=datamodule.worker_skills,
        n_epoch=args.n_epoch,
        n_epoch_batches=len(dummy_loader),
        n_workers=len(datamodule.worker_mapping.keys()),
        num_labels=len(datamodule.label_mapping),
        lr=learning_rate,
        label_smoothing=args.label_smoothing,
        fixmatch_coef=args.fixmatch_coef,
        fixmatch_threshold=args.fixmatch_threshold,
        fixmatch_softmax_t=args.fixmatch_softmax_t,
        dropout=args.dropout,
    )

    with torch.no_grad():
        model(**next(iter(dummy_loader)))

    trainer = pl.Trainer(
        max_epochs=args.n_epoch,
        gradient_clip_val=1,
        terminate_on_nan=True,
        accelerator=args.accelerator,
        gpus=args.gpus,
        # auto_scale_batch_size="binsearch",
        # accumulate_grad_batches=2,
    )

    mlflow_tracking_uri = None
    if args.mlflow:
        import mlflow
        from annotator_model.sweep_utils import _get_workspace

        ws, experiment = _get_workspace()
        mlflow_tracking_uri = ws.get_mlflow_tracking_uri()
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        if not experiment:
            mlflow.set_experiment(args.experiment_name)

        mlflow.log_params(vars(args))
        mlflow.log_params(hparams_to_dict(model.hparams))
        mlflow.log_params(hparams_to_dict(datamodule.hparams))

        mlflow.pytorch.autolog(log_every_n_step=300)

    if args.wandb:
        import os

        if args.wandb_api_key:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
        import wandb

        wandb.init(
            project=args.experiment_name,
            entity="toloka-research",
            config=args,
        )

    callbacks, trainer, skills_callback = model.configure_trainer(
        trainer,
        mlflow=args.mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        wandb=args.wandb,
        recompute_worker_skills=args.recompute_worker_skills,
        worker_skill_method=args.worker_skill_method,
        start_recompute_skills_after_epoch=args.start_recompute_skills_after_epoch,
    )

    # Train
    trainer.fit(
        model,
        datamodule,
    )

    # Test
    test_results = trainer.test(model, datamodule.test_dataloader())
    print(f"Test results:\n{test_results}")

    time_curr = time.time()
    time_elapsed = time_curr - time_start
    time_elapsed_str = f"Time elapsed: {time_elapsed//3600:.0f}h {(time_elapsed%3600)//60:.0f}m {(time_elapsed%3600)%60:.0f}s"

    print(time_elapsed_str)
    print("Done")

    del datamodule.trainer

    if skills_callback:
        with open("worker_skills_history.pkl", "wb") as f:
            pickle.dump(skills_callback.skills_history, f)

    with open("datamodule.pkl", "wb") as f:
        pickle.dump(datamodule, f)

    if args.mlflow:
        mlflow.log_text(time_elapsed_str, "time_elapsed")
        mlflow.log_artifact("datamodule.pkl")
        if skills_callback:
            mlflow.log_artifact("worker_skills_history.pkl")

    if args.wandb:
        wandb.log_artifact("datamodule.pkl", type="datamodule")
        if skills_callback:
            wandb.log_artifact("worker_skills_history.pkl", type="extra")


if __name__ == "__main__":
    main()
