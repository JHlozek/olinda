"""Model Tuner."""

from abc import ABC, abstractmethod

from random import random
from typing import Any, List
import shutil
import os
import copy

import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from olinda.data import GenericOutputDM, XgboostDataset
from olinda.generic_model import GenericModel

class ModelTuner(ABC):
    """Automatic model tuner."""

    @abstractmethod
    def fit(self: "ModelTuner", datamodule: GenericOutputDM) -> GenericModel:
        """Fit an optimal model using the given dataset.

        Args:
            datamodule (GenericOutputDM): Datamodule to fit an optimal model.

        Returns:
            GenericModel : Student model as wrapped in a generic model class.
        """
        pass

class XgboostTuner(ModelTuner):
    """XGBoost based model tuner."""

    def __init__(
        self: "XgboostTuner",
    ) -> None:
        """Initialize model tuner.

        Args:
            
        """
        pass

    def fit(self: "XgboostTuner", datamodule: GenericOutputDM, time_budget=1800) -> GenericModel:
        """Fit an optimal model using the given dataset.

        Args:
            datamodule (GenericOutputDM): Datamodule to fit an optimal model.
        Returns:
            GenericModel : Student model as wrapped in a generic model class.
            time_budget (int): Hyperparameter search time allowance
        """
        
        self.datamodule = datamodule
        train_dataset = XgboostDataset(self.datamodule, "train")
        
        self._hyperparam_search(train_dataset, time_budget=time_budget)
        self._final_train(train_dataset)
        return GenericModel(self.model)
    """
    def find_GPUs(self):
        GPUs = GPUtil.getGPUs()
        if len(GPUs) > 0 and TRY_GPU:
            return "GPU"
        return "CPU"
    """

    def objective(self, trial):
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            self.X, self.y, self.weights, test_size=0.2, random_state=42
        )
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        param = {
            "silent": 1,
            "objective": "reg:linear",
            "eval_metric": "mae",
            "booster": "gbtree",
            #"early_stopping_rounds": 100, 
            "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        }

        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
        param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-mae")
        reg = xgb.train(param, dtrain, evals=[(dtest, "validation")], callbacks=[pruning_callback])
        y_pred = reg.predict(dtest)
        score = mean_absolute_error(y_test, y_pred)
        return score

    def _hyperparam_search(self: "XgboostTuner", train_dataset: XgboostDataset, time_budget=1800):
        print("Starting hyperparameter search for", time_budget, "seconds.")

        self.X = train_dataset.X
        self.y = train_dataset.y
        self.weights = train_dataset.weights

        self.study = optuna.create_study(sampler=TPESampler(), direction="minimize")

        self.study.optimize(self.objective, n_trials=500, timeout=time_budget, gc_after_trial=True)
        self.best_params = self.study.best_trial.params
        print("Best trial parameters:", self.best_params)
       
        
    def _final_train(self: "XgboostTuner", train_dataset: XgboostDataset):
        dtrain = xgb.DMatrix(train_dataset.X, label=train_dataset.y, weight=train_dataset.weights)        
        self.model = xgb.train(self.best_params, dtrain)
        return GenericModel(self.model)

