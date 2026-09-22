import logging
import os
import socket
from statistics import mean, stdev
from typing import Any, Dict

import mlflow


def setup_mlflow(experiment_name: str) -> None:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    prepare_mlflow_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)


def prepare_mlflow_experiment(exp_name: str) -> None:
    if not mlflow.get_experiment_by_name(exp_name):
        create_mlflow_experiment(exp_name)


def create_mlflow_experiment(exp_name: str) -> None:
    tags: Dict[str, Any] = {
        "project_name": "QMSC_Late_Fusion",
    }

    mlflow.create_experiment(
        name=exp_name,
        tags=tags,
        artifact_location=os.environ["MLFLOW_ARTIFACTS_ROOT"],
    )


def log_params(params: Dict[str, Any]) -> None:
    mlflow.set_tag("hostname", socket.gethostname())
    mlflow.set_tag("model", params["experiment_params"]["model_name"])
    for _, value in params.items():
        log_nested_params(value)


def log_nested_params(params: Dict[str, Any]) -> None:
    # Remove duplicate, if they happend to
    for k in mlflow.get_run(
        mlflow.active_run().info.run_id
    ).data.params.keys():  # type: ignore
        if k in params.keys():
            logging.warning(
                f"Parameter {k} already logged, skipping duplicate."
            )
            params.pop(k)

    mlflow.log_params(params)


def start_parent_run(model_name: str) -> mlflow.ActiveRun:
    run = mlflow.start_run(run_name=model_name)
    return run


def start_child_hp_run(fold_name: str) -> mlflow.ActiveRun:
    return mlflow.start_run(run_name=fold_name, nested=True)


def log_metrics(metrics: Dict[str, Any]) -> None:
    for metric_name, values in metrics.items():
        mlflow.log_metric(metric_name, values)


def log_aggregated_metrics(
    all_fold_metrics: dict, preds: list, true_labels: list
) -> None:
    for metric_name, values in all_fold_metrics.items():
        mlflow.log_metric(f"{metric_name}_mean", mean(values))
        mlflow.log_metric(f"{metric_name}_std", stdev(values))


# def log_model(
#     trainer: BaseTraining,
#     model: BaseMLPModel,
#     dataset: BaseDataset,
#     model_name: str = "model",
# ) -> None:
#     signature = infer_signature(
#         dataset.val_data, trainer.predict(model, dataset)
#     )
#     trainer.log_model(model=model, model_name=model_name, signature=signature)
