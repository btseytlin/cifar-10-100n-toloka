import pandas as pd
import torch
from crowdkit.aggregation import MajorityVote, ZeroBasedSkill
from crowdkit.metrics.workers import accuracy_on_aggregates
from torch import nn
from torch.nn.modules.lazy import LazyModuleMixin
from sklearn.metrics import accuracy_score
from pytorch_lightning.utilities import AttributeDict
import numpy as np
import random
import torch
import torch.nn.functional as F

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)


def make_toloka_format_csvs(
    side_info_df_path="./data/side_info_cifar10N.csv",
    noisy_labels_dict_path="./data/CIFAR-10_human.pt",
    labels_df_output_path="./data/labels.csv",
    toloka_df_output_path="./data/train_toloka.csv",
):
    side_info_df = pd.read_csv(side_info_df_path)
    noisy_labels_dict = torch.load(noisy_labels_dict_path)
    labels_df = pd.DataFrame(noisy_labels_dict)
    labels_df["img_index"] = range(len(labels_df))
    toloka_format_rows = []
    for row in side_info_df.itertuples():
        batch_index_start = int(row[1].split("--")[0])
        batch_index_end = int(row[1].split("--")[1])
        img_indices = range(batch_index_start, batch_index_end + 1)

        for idx in img_indices:
            worker_1_id = row[2]
            worker_1_label = noisy_labels_dict["random_label1"][idx]

            worker_2_id = row[4]
            worker_2_label = noisy_labels_dict["random_label2"][idx]

            worker_3_id = row[6]
            worker_3_label = noisy_labels_dict["random_label3"][idx]

            toloka_format_rows.append(
                {
                    "task": idx,
                    "worker": worker_1_id,
                    "label": worker_1_label,
                    "source": "random_label1",
                }
            )
            toloka_format_rows.append(
                {
                    "task": idx,
                    "worker": worker_2_id,
                    "label": worker_2_label,
                    "source": "random_label2",
                }
            )
            toloka_format_rows.append(
                {
                    "task": idx,
                    "worker": worker_3_id,
                    "label": worker_3_label,
                    "source": "random_label3",
                }
            )
    train_toloka_df = pd.DataFrame.from_records(toloka_format_rows)
    labels_df.to_csv(labels_df_output_path, index=False)
    train_toloka_df.to_csv(toloka_df_output_path, index=False)
    return labels_df_output_path, toloka_df_output_path


def aggregation_skills(df, aggregator=None):
    aggregator = aggregator or MajorityVote()
    skills = accuracy_on_aggregates(answers=df, by="worker", aggregator=aggregator)
    return skills


class LazyLinearIdentical(LazyModuleMixin, nn.Linear):
    cls_to_become = nn.Linear  # type: ignore[assignment]
    weight: nn.parameter.UninitializedParameter
    bias: nn.parameter.UninitializedParameter  # type: ignore[assignment]

    def __init__(self, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # bias is hardcoded to False to avoid creating tensor
        # that will soon be overwritten.
        super().__init__(0, 0, False)
        self.weight = nn.parameter.UninitializedParameter(**factory_kwargs)
        if bias:
            self.bias = nn.parameter.UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.in_features = input.shape[-1]
                self.weight.materialize((self.in_features, self.in_features))
                if self.bias is not None:
                    self.bias.materialize((self.in_features,))
                self.reset_parameters()


def get_mlp(output_dim, dropout_rate):
    return nn.Sequential(
        nn.Dropout(dropout_rate),
        LazyLinearIdentical(),
        nn.Tanh(),
        nn.LazyLinear(output_dim),
    )


def batch_to_device(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch


def hparams_to_dict(hparams):
    new_dict = dict(**hparams)
    for k, v in new_dict.items():
        if isinstance(v, AttributeDict):
            new_dict[k] = hparams_to_dict(v)
    return new_dict


def set_global_seeds(i):
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(i)


def set_device():
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")
    print(f"Current device is {_device}", flush=True)
    return _device


# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, epoch, alpha_plan):
    for param_group in optimizer.param_groups:
        param_group["lr"] = alpha_plan[epoch]


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def bernoulli_bayes_est(correct, total, alpha=5, beta=2):
    return (alpha + correct) / (alpha + beta + total)


def accuracy_beta_prior(true, pred):
    correct = np.sum(true == pred)
    return bernoulli_bayes_est(correct=correct, total=len(true))


def compute_worker_metrics(df, metric=accuracy_score, true_label_col="true_label"):
    metrics = []
    for worker in df.worker.unique():
        subbf = df[df.worker == worker]
        metric_val = metric(subbf[true_label_col], subbf["label"])
        metrics.append({"worker": worker, "metric": metric_val})
    result = pd.DataFrame.from_records(metrics).set_index("worker")["metric"]
    return result


def compute_worker_skills(df, worker_skill_method, true_labels_included=False):
    if worker_skill_method == "mv":
        skills = aggregation_skills(df, aggregator=MajorityVote())
    elif worker_skill_method == "zbs":
        skills = aggregation_skills(df, aggregator=ZeroBasedSkill())
    elif worker_skill_method == "mv_prior":
        df_copy = df.copy()
        if not true_labels_included:
            mv_labels = MajorityVote().fit_predict(df).to_dict()
            df_copy["true_label"] = df_copy.task.apply(lambda t: mv_labels[t])
        skills = compute_worker_metrics(df_copy, metric=accuracy_beta_prior)
    return skills
