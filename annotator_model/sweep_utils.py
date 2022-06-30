from azureml.core import (
    Workspace,
    Run,
    Dataset,
    ScriptRunConfig,
    Experiment,
    Environment,
)


def _get_workspace():
    run = Run.get_context()
    if hasattr(run, "experiment"):
        # running in a compute cluster
        return run.experiment.workspace, run.experiment
    else:
        # running from a compute instance (dev container)
        # TODO: this assumes config.json in the root, which is going to be changed
        return Workspace.from_config(), None


def get_mlflow_tracking_uri():
    ws, _ = _get_workspace()
    return ws.get_mlflow_tracking_uri()


def submit(
    experiment_name: str,
    command: str,
    compute_target: str,
    environment: str,
    env_variables: dict = None,
):
    """
    Submit a new experiment as a job on a cluster.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    command : str
        Command to run.
    arguments : list, optional
        Arguments to the script.
    compute_target : str, optional
        Name of the cluster to run the script on.
    environment : str, optional
        Name of the environment (e.g. Docker env) to run the script in.

    Examples
    --------

    >>> submit("my-text-experiment", "toloka/ah/text.py", [...])
    """

    ws = _get_workspace()
    # create or load an experiment
    experiment = Experiment(ws, experiment_name)
    # create or retrieve a compute target
    cluster = ws.compute_targets[compute_target]
    # create or retrieve an environment
    env = Environment.get(ws, name=environment)
    # for key, e in Environment.list(ws).items():
    #     if key == 'ce-docker':
    #         print('Fuck')
    #         env = e

    # env = Environment("ce-environment")
    # env.docker.base_dockerfile = "dockerfile"

    # env.register(ws).build(ws).wait_for_completion()

    # set environment variables
    env = env.clone("customize_curated")
    if env_variables is not None:
        env.environment_variables.update(env_variables)
    # print(env)
    # print(ws)
    # configure and submit your training run
    config = ScriptRunConfig(
        source_directory=".",
        command=command.split(),
        compute_target=cluster,
        environment=env,
    )
    script_run = experiment.submit(config)
    print(
        f"Experiment {experiment_name} has been submitted. "
        + "Please, see AzureML UI for details"
    )


def submit_sweep_iteration(
    experiment_name: str,
    sweep_id: str,
    compute_target: str,
    environment: str,
    wandb_api_key: str = None,
    iterations: int = 1,
) -> None:
    """
    Submit a new sweep iteration as a job on a cluster.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    sweep_id : str
        ID of the sweep to run.
    compute_target : str, optional
        Name of the cluster to run the script on.
    environment : str, optional
        Name of the environment (e.g. Docker env) to run the script in.
    wandb_api_key : str, optional
        API key for W&B.

    Examples
    --------

    >>> submit_sweep_iteration("my-text-experiment", "sweep-id", "V100-1x", "ce-environment")
    """

    submit(
        experiment_name=experiment_name,
        command=f"pip uninstall --yes tensorflow-gpu apex && pip install -r requirements.txt && pip install -e . && wandb agent --count {iterations} {sweep_id}",
        compute_target=compute_target,
        environment=environment,
        env_variables={"WANDB_API_KEY": wandb_api_key},
    )


def submit_sweep_iterations(
    experiment_name: str,
    sweep_id: str,
    compute_target: str,
    environment: str,
    wandb_api_key: str = None,
    n_jobs: int = 20,
    iterations_per_job: int = 1,
) -> None:
    """
    Submit multiple sweep iterations as jobs on a cluster.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    sweep_id : str
        ID of the sweep to run.
    compute_target : str, optional
        Name of the cluster to run the script on.
    environment : str, optional
        Name of the environment (e.g. Docker env) to run the script in.
    wandb_api_key : str, optional
        API key for W&B.
    n_jobs: int, optional
        Number of sweep jobs to run.
    iterations_per_job : int, optional
        Number of iterations to run per job.

    Examples
    --------

    >>> submit_sweep_iterations("my-text-experiment", "sweep-id", "V100-1x", "ce-environment")
    """

    for i in range(n_jobs):
        submit_sweep_iteration(
            experiment_name=experiment_name,
            sweep_id=sweep_id,
            compute_target=compute_target,
            environment=environment,
            wandb_api_key=wandb_api_key,
            iterations=iterations_per_job,
        )
