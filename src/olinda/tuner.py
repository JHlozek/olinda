"""Model Tuner."""

from abc import ABC, abstractmethod

from random import random
from typing import Any, List
import shutil
import os
import copy

import autokeras as ak
import kerastuner as kt
import tensorflow as tf
from tensorflow import keras

import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from olinda.data import GenericOutputDM, XgboostDataset, TensorflowDatasetWrapper
from olinda.generic_model import GenericModel

# overwrite checkpoint functionality
class HyperbandTuner(kt.Hyperband):
    def __init__(self, hypermodel, **kwargs):
        super().__init__(hypermodel, **kwargs)
    
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        model = self.hypermodel.build(hp)
        return self.hypermodel.fit(hp, model, *args, **kwargs)

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


class AutoKerasTuner(ModelTuner):
    """AutoKeras based model tuner."""

    def __init__(self: "AutoKerasTuner", max_trials: int = 3) -> None:
        """Initialize model tuner.

        Args:
            max_trials (int): Maximum interations to perform.
        """
        self.max_trials = max_trials

    def fit(self: "AutoKerasTuner", datamodule: GenericOutputDM) -> GenericModel:
        """Fit an optimal model using the given dataset.

        Args:
            datamodule (GenericOutputDM): Datamodule to fit an optimal model.

        Returns:
            GenericModel : Student model as wrapped in a generic model class.
        """
        self.mdl = ak.StructuredDataRegressor(
            overwrite=False,
            max_trials=self.max_trials,
            project_name=f"autokeras-{random()*1000}",
        )
        tensor_wrapper = TensorflowDatasetWrapper(
            datamodule, "train", only_X=True, only_Y=True, weights=True
        )
        self.dataset = tf.data.Dataset.from_generator(
            generator=tensor_wrapper.__iter__,
            output_signature=(
                tf.TensorSpec(shape=(1024,), dtype=tf.float32),
                tf.TensorSpec(shape=(1,), dtype=tf.float32),
                tf.TensorSpec(shape=(1,), dtype=tf.float32),
            ),
        )
        self.mdl.fit(self.dataset)
        return GenericModel(self.mdl.export_model())

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
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        param = {
            "silent": 1,
            "objective": "reg:linear",
            "eval_metric": "mae",
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        }

        if param["booster"] == "gbtree" or param["booster"] == "dart":
            param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
            param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
            param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

        reg = xgb.XGBRegressor(**param)
        reg.fit(dtrain, eval_set=dtest, sample_weight=weights_train, early_stopping_rounds=100, verbose=0)
        y_pred = reg.predict(X_test)
        score = mean_absolute_error(y_test, y_pred)
        return score

    def _hyperparam_search(self: "XgboostTuner", train_dataset: XgboostDataset, time_budget=1800):
        print("Starting hyperparameter search for", time_budget, "seconds.")

        self.X = train_dataset.X
        self.y = train_dataset.y
        self.weights = train_dataset.weights

        self.study = optuna.create_study(sampler=TPESampler(), direction="minimize")

        self.study.optimize(self.objective, n_trials=500, timeout=time_budget)
        self.best_params = self.study.best_trial.params
        print("Best trial parameters:", self.best_params)
       
        
    def _final_train(self: "XgboostTuner", train_dataset: XgboostDataset):
        dtrain = xgb.DMatrix(train_dataset.X, label=train_dataset.y)        
        if self.best_params is not None:
            self.model = xgb.XGBRegressor(**self.best_params)
        else:
            self.model = xgb.XGBRegressor()
        self.model.fit(dtrain, sample_weight=train_dataset.weights, early_stopping_rounds=100)


class KerasTuner(ModelTuner):
    """Keras tuner based model tuner."""

    def __init__(
        self: "KerasTuner", layers_range: List = [2, 4], max_epochs: int = 30
    ) -> None:
        """Initialize model tuner.

        Args:
            layers_range (List): Range of hidden layers to search.
        """
        self.layers_range = layers_range
        self.max_epochs = max_epochs
        self.input_shape = 1024

    def fit(self: "KerasTuner", datamodule: GenericOutputDM) -> GenericModel:
        """Fit an optimal model using the given dataset.

        Args:
            datamodule (GenericOutputDM): Datamodule to fit an optimal model.

        Returns:
            GenericModel : Student model as wrapped in a generic model class.
        """
        
        self.datamodule = datamodule
        self.search_datamodule = copy.deepcopy(self.datamodule)
        train_search_dataset = self._tensor_wrapper(self.search_datamodule, "train", True)
        val_search_dataset = self._tensor_wrapper(self.search_datamodule, "val", True)
        
        train_dataset = self._tensor_wrapper(self.datamodule, "train")
        val_dataset = self._tensor_wrapper(self.datamodule,"val")
        tensor_wrapper = TensorflowDatasetWrapper(self.datamodule, "train", only_X=True, only_Y=True, weights=True)
        self.output_shape = tensor_wrapper.output_signature()[1].shape[0] #Length of output tensor
               
        self._search(train_search_dataset, val_search_dataset)
        self._get_best_epoch(train_dataset, val_dataset)
        self._final_train(train_dataset, val_dataset)
        return GenericModel(self.hypermodel)

    def _tensor_wrapper(self: "KerasTuner", datamodule: GenericOutputDM, stage: str, smaller_set: bool = False):
        stages_allowed = ["train", "val"]
        assert stage in stages_allowed
        
        tensor_wrapper = TensorflowDatasetWrapper(
            datamodule, stage, only_X=True, only_Y=True, weights=True, smaller_set=smaller_set
        )
        
        dataset = tf.data.Dataset.from_generator(
            generator=tensor_wrapper.__iter__,
            output_signature=tensor_wrapper.output_signature(),
        )
        
        return dataset.batch(32)

    def _model_builder(self: "KerasTuner", hp: Any):
        model = keras.Sequential()
        hp_units = hp.Int("first_layer_units", min_value=32, max_value=512, step=32)
        model.add(keras.layers.Flatten())
        model.add(
            keras.layers.Dense(
                units=hp_units, activation="relu", input_shape=(self.input_shape,)
            )
        )
        for i in range(hp.Int("layers", self.layers_range[0], self.layers_range[1])):
            model.add(
                keras.layers.Dense(
                    units=hp.Int(
                        "hidden_units_" + str(i), min_value=32, max_value=512, step=32
                    ),
                    activation="relu",
                )
            )
        model.add(keras.layers.Dense(self.output_shape))
        # Tune the learning rate for the optimizer
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss="mean_squared_error",
            weighted_metrics=[],
        )

        return model

    def _search(
        self: "KerasTuner", train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset
    ) -> None:
        print("Hyperparameter Search")
        
        if os.path.exists("trials"):
            shutil.rmtree("trials")
        
        self.tuner = HyperbandTuner(
            self._model_builder,
            objective="val_loss",
            max_epochs=3,
            factor=3,
            project_name="trials",
        )
        stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
        self.tuner.search(
            train_dataset,
            epochs=3,
            validation_data=val_dataset,
            callbacks=[stop_early],
            verbose=True,
        )
        self.best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        shutil.rmtree("trials")

    def _get_best_epoch(
        self: "KerasTuner", train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset
    ) -> None:
        print("Best Epoch Search")
        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        model = self.tuner.hypermodel.build(self.best_hps)
        history = model.fit(
            train_dataset, epochs=self.max_epochs, validation_data=val_dataset
        )

        val_per_epoch = history.history["val_loss"]
        self.best_epoch = val_per_epoch.index(min(val_per_epoch)) + 1
        print("Best epoch: %d" % (self.best_epoch,))
        self.hypermodel = model

    def _final_train(
        self: "KerasTuner", train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset
    ):
        print("Final Model")
        self.hypermodel = self.tuner.hypermodel.build(self.best_hps)

        # Retrain the model
        self.hypermodel.fit(
            train_dataset, epochs=self.best_epoch, validation_data=val_dataset
        )
