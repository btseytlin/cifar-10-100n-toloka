import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from crowdkit.aggregation import MajorityVote, ZeroBasedSkill
from torch.utils.data import DataLoader
from torchvision import transforms

from annotator_model.dataset import (
    ImageDataset,
    ImageDatasetFromPickle,
    FixMatchImageDataset,
)
from annotator_model.fixmatch import TransformFixMatch
from annotator_model.utils import (
    aggregation_skills,
    cifar10_mean,
    cifar10_std,
    compute_worker_skills,
)
from annotator_model.data.utils import download_url, check_integrity


default_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ]
)


class CIFAR10DataModule(pl.LightningDataModule):
    label_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_compressed_image_batches = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_compressed_image_batches = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]

    def __init__(
        self,
        root="./data/cifar10_data/",
        labels_csv_path="./data/labels.csv",
        toloka_csv_path="./data/train_toloka.csv",
        download=False,
        noise_type=None,
        input_size=224,
        worker_mapping=None,
        worker_skill_method="zbs",
        val_ratio=0.1,
        batch_size=32,
        num_workers=4,
        train_transform=None,
        val_transform=None,
        debug_limit=None,
    ):
        super().__init__()
        self.root = os.path.expanduser(root)

        self.noise_type = noise_type
        self.labels_csv_path = labels_csv_path
        self.toloka_csv_path = toloka_csv_path

        self.download = download

        self.input_size = input_size

        self.worker_mapping = worker_mapping
        self.worker_skill_method = worker_skill_method
        self.val_ratio = val_ratio

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform or default_transform
        self.val_transform = val_transform or default_transform

        self.debug_limit = debug_limit

        self.worker_skills = None
        self.train_images = None
        self.test_images = None
        self.train_df = None
        self.val_df = None
        self.train = None
        self.train_fixmatch = None
        self.val = None
        self.test = None

        self._is_setup = False

    def _check_integrity(self):
        root = self.root
        for fentry in (
            self.train_compressed_image_batches + self.test_compressed_image_batches
        ):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download_dataset(self):
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def get_df_for_noise_type(self):
        toloka_df = pd.read_csv(self.toloka_csv_path)
        if self.debug_limit:
            debug_tasks = np.random.choice(
                toloka_df.task.unique(), size=self.debug_limit
            )
            toloka_df = toloka_df[toloka_df.task.isin(debug_tasks)]
        if self.noise_type == "aggre_label":
            return toloka_df
        elif self.noise_type == "random_label1":
            return toloka_df[toloka_df.source == "random_label1"]
        elif self.noise_type == "worse_label":
            labels_df = pd.read_csv(self.labels_csv_path)
            worst_task_labels = labels_df.set_index("img_index")[
                "worse_label"
            ].to_dict()

            new_rows = []
            for row in toloka_df.itertuples():
                new_rows.append(
                    dict(
                        task=row.task,
                        worker=row.worker,
                        label=worst_task_labels[row.task],
                    )
                )
            new_df = pd.DataFrame.from_records(new_rows)
            return new_df
        else:
            raise NotImplementedError(f"Noise type {self.noise_type} not supported.")

    def get_worker_mapping(self, *dfs):
        workers = set()
        for df in dfs:
            if df is not None and "worker" in df.columns:
                workers = workers.union(df.worker.unique())
        return {w: i for i, w in enumerate(workers)}

    def load_datasets(self, train_df, val_df):
        self.train_images = []
        for fentry in self.train_compressed_image_batches:
            f = fentry[0]
            file = os.path.join(self.root, self.base_folder, f)
            with open(file, "rb") as fo:
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding="latin1")
                self.train_images.append(entry["data"])

        self.train_images = np.concatenate(self.train_images)
        self.train_images = self.train_images.reshape((50000, 3, 32, 32))
        self.train_images = self.train_images.transpose((0, 2, 3, 1))

        train_dataset = ImageDataset.get_dataset_from_df(
            train_df,
            self.train_images,
            self.label_mapping,
            self.worker_mapping,
            transform=self.train_transform,
        )

        train_fixmatch_dataset = FixMatchImageDataset.get_dataset_from_df(
            train_df,
            self.train_images,
            self.label_mapping,
            self.worker_mapping,
            input_size=self.input_size,
            transform=TransformFixMatch(
                input_size=self.input_size, mean=cifar10_mean, std=cifar10_std
            ),
        )

        val_dataset = ImageDataset.get_dataset_from_df(
            val_df,
            self.train_images,
            self.label_mapping,
            self.worker_mapping,
            transform=self.val_transform,
        )

        f = self.test_compressed_image_batches[0][0]
        file = os.path.join(self.root, self.base_folder, f)
        test_dataset = ImageDatasetFromPickle(file, transform=self.val_transform)
        return train_dataset, train_fixmatch_dataset, val_dataset, test_dataset

    def compute_and_set_worker_skills(self, df):
        skills = compute_worker_skills(df, self.worker_skill_method)

        self.worker_skills = pd.Series(
            {
                self.worker_mapping[worker_id]: skill
                for worker_id, skill in dict(skills).items()
            }
        )
        self.worker_skills = self.worker_skills.fillna(0)

    def train_val_split(self, df):
        training_tasks = df.task.unique()
        num_training_tasks = len(training_tasks)
        task_idx_full = np.arange(num_training_tasks)
        np.random.shuffle(task_idx_full)
        train_tasks = training_tasks[
            task_idx_full[int(num_training_tasks * self.val_ratio) :]
        ]
        val_tasks = training_tasks[
            task_idx_full[: int(num_training_tasks * self.val_ratio)]
        ]

        train_df, val_df = (
            df[df.task.isin(train_tasks)],
            df[df.task.isin(val_tasks)],
        )
        return train_df, val_df

    def setup(self, stage=None):
        if self._is_setup:
            return

        if self.download:
            self.download_dataset()
        elif not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        toloka_df = self.get_df_for_noise_type()

        train_df, val_df = self.train_val_split(toloka_df)

        self.train_df = train_df
        self.val_df = val_df

        self.worker_mapping = self.get_worker_mapping(train_df, val_df)
        self.compute_and_set_worker_skills(train_df)
        self.train, self.train_fixmatch, self.val, self.test = self.load_datasets(
            train_df, val_df
        )

        self._is_setup = True

    @property
    def inverse_label_mapping(self):
        return {v: k for k, v in enumerate(self.label_mapping)}

    def train_dataloader(self, shuffle=True):
        collate = getattr(self.train, "collate", None)

        return [
            DataLoader(
                self.train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=shuffle,
                collate_fn=collate,
            ),
            DataLoader(
                self.train_fixmatch,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=shuffle,
                collate_fn=collate,
            ),
        ]

    def val_dataloader(self):
        collate = getattr(self.val, "collate", None)
        return [
            DataLoader(
                self.val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate,
            ),
            self.test_dataloader(),
        ]

    def test_dataloader(self):
        collate = getattr(self.test, "collate", None)
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate,
        )
