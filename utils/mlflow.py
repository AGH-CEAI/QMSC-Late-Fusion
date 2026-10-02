import logging
import os
import socket
from typing import Any, Dict

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature


################################################################################
# MLFLOW SETUP #################################################################
################################################################################
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


################################################################################
# MLFLOW PARAMS ################################################################
################################################################################
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


################################################################################
# MLFLOW START RUN #############################################################
################################################################################


def start_parent_run(model_name: str) -> mlflow.ActiveRun:
    run = mlflow.start_run(run_name=model_name)
    return run


def start_child_hp_run(fold_name: str) -> mlflow.ActiveRun:
    return mlflow.start_run(run_name=fold_name, nested=True)


################################################################################
# MLFLOW METRICS ###############################################################
################################################################################
def log_metrics(results: Dict[str, Any]) -> None:
    metrics = {
        "mean_fit_time": np.mean(results["fit_time"]),
        "mean_score_time": np.mean(results["score_time"]),
    }

    for key, values in results.items():
        if key.startswith("test_"):
            name = key.replace("test_", "")

            # Mean
            metrics[f"mean_{name}"] = np.mean(values)

            # Per Fold
            for fold_idx, val in enumerate(values):
                metrics[f"{name}_fold_{fold_idx}"] = val


################################################################################
# MLFLOW MODEL #################################################################
################################################################################


def log_sk_model(model, model_name, data):
    signature = infer_signature(data, model.predict(data))
    mlflow.sklearn.log_model(
        sk_model=model, name=model_name, signature=signature
    )
