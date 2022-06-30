from argparse import ArgumentParser
from annotator_model.sweep_utils import submit_sweep_iterations


def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--compute_target", type=str, default="V100-1x")
    parser.add_argument(
        "--environment",
        type=str,
        default="AzureML-pytorch-1.10-ubuntu18.04-py38-cuda11-gpu",
    )
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--n_jobs", type=int, default=20)
    parser.add_argument("--iterations_per_job", type=int, default=1)
    args = parser.parse_args()

    submit_sweep_iterations(
        experiment_name=args.experiment_name,
        sweep_id=args.sweep_id,
        compute_target=args.compute_target,
        environment=args.environment,
        wandb_api_key=args.wandb_api_key,
        n_jobs=args.n_jobs,
        iterations_per_job=args.iterations_per_job,
    )


if __name__ == "__main__":
    main()
