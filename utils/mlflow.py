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


def log_params(config: Dict[str, Any]) -> None:
    def log_nested_params(conf_dict, prefix=""):
        for key, value in conf_dict.items():
            if isinstance(value, dict):
                log_nested_params(value, f"{prefix}{key}.")
            else:
                mlflow.log_param(key=f"{prefix}{key}", value=value)

    mlflow.set_tag("hostname", socket.gethostname())
    mlflow.set_tag("model_name", config["model_name"])
    log_nested_params(config)


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
def log_cross_val_metrics(score: Dict[str, Any]) -> None:
    metrics = {}
    for key, values in score.items():
        if key.startswith("test_"):
            name = key.replace("test_", "")
        else:
            name = key

        # Mean
        metrics[f"mean_{name}"] = np.mean(values)

        # Per Fold
        for fold_idx, val in enumerate(values):
            metrics[f"{name}_fold_{fold_idx + 1}"] = val
    mlflow.log_metrics(metrics=metrics)


################################################################################
# MLFLOW MODEL #################################################################
################################################################################


def log_sk_model(model, model_name, data):
    signature = infer_signature(data, model.predict(data))
    mlflow.sklearn.log_model(
        sk_model=model, name=model_name, signature=signature
    )
