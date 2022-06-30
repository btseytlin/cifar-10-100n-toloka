import pickle

import torch
from PIL import Image
from torch.utils.data import Sampler

from annotator_model.utils import cifar10_mean, cifar10_std
from annotator_model.fixmatch import TransformFixMatch


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels,
        workers=None,
        task_ids=None,
        transform=None,
    ):
        super().__init__()
        self.images = images
        self.workers = torch.LongTensor(workers) if workers is not None else None
        self.task_ids = task_ids
        self.labels = torch.LongTensor(labels)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index], "RGB")
        if self.transform is not None:
            img = self.transform(img)

        labels = self.labels[index]

        workers = None
        if self.workers is not None:
            workers = self.workers[index]

        task_ids = None
        if self.task_ids is not None:
            task_ids = self.task_ids[index]

        return dict(images=img, labels=labels, workers=workers, task_ids=task_ids)

    @classmethod
    def get_dataset_from_df(cls, df, images, label_mapping, worker_mapping, **kwargs):
        agg_dict = {"label": list, "worker": list}
        grouped_df = df.groupby("task").agg(agg_dict).reset_index()
        task_ids = img_indices = grouped_df.task.values
        images = images[img_indices]
        labels = grouped_df.label.values
        labels = [[label_mapping[x] for x in inner_list] for inner_list in labels]

        workers = grouped_df.worker.values
        workers = [[worker_mapping[x] for x in inner_list] for inner_list in workers]

        return cls(
            images=images, labels=labels, workers=workers, task_ids=task_ids, **kwargs
        )


class FixMatchImageDataset(ImageDataset):
    def __init__(
        self,
        images,
        labels,
        input_size,
        workers=None,
        task_ids=None,
        transform=None,
    ):
        super().__init__(
            images=images, labels=labels, workers=workers, task_ids=task_ids
        )
        self.input_size = input_size
        self.transform = transform or TransformFixMatch(
            mean=cifar10_mean,
            std=cifar10_std,
            input_size=input_size,
        )

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index], "RGB")
        img_weak, img_strong = self.transform(img)

        labels = self.labels[index]

        workers = None
        if self.workers is not None:
            workers = self.workers[index]

        task_ids = None
        if self.task_ids is not None:
            task_ids = self.task_ids[index]

        return dict(
            images_weak=img_weak,
            images_strong=img_strong,
            labels=labels,
            workers=workers,
            task_ids=task_ids,
        )


class ImageDatasetFromPickle(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        super(ImageDatasetFromPickle, self).__init__()
        data = self.unpickle(path)
        self.imgs = data[b"data"].reshape((len(data[b"data"]), 3, 32, 32))
        self.imgs = self.imgs.transpose(0, 2, 3, 1)
        self.transform = transform
        self.labels = torch.LongTensor(data[b"labels"])

    def __getitem__(self, index):
        img = Image.fromarray(self.imgs[index], "RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"images": img, "labels": self.labels[index]}

    def __len__(self):
        return len(self.imgs)

    def unpickle(self, file):
        with open(file, "rb") as fo:
            dict_ = pickle.load(fo, encoding="bytes")
        return dict_
