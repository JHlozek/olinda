"""Model Tuner."""

from abc import ABC, abstractmethod
from random import random
from typing import Any, List

import autokeras as ak
import kerastuner as kt
import tensorflow as tf
from tensorflow import keras

from olinda.data import GenericOutputDM
from olinda.generic_model import GenericModel


class ModelTuner(ABC):
    """Automatic model tuner."""

    @abstractmethod
    def fit(self: "ModelTuner", dataset: GenericOutputDM) -> GenericModel:
        """Fit an optimal model using the given dataset.

        Args:
            dataset (GenericOutputDM): Dataset to fit an optimal model.

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

    def fit(self: "AutoKerasTuner", dataset: GenericOutputDM) -> GenericModel:
        """Fit an optimal model using the given dataset.

        Args:
            dataset (GenericOutputDM): Dataset to fit an optimal model.

        Returns:
            GenericModel : Student model as wrapped in a generic model class.
        """
        self.mdl = ak.StructuredDataRegressor(
            overwrite=False,
            max_trials=self.max_trials,
            project_name=f"autokeras-{random()*1000}",
        )
        self.X = dataset.dataset[2]
        self.Y = dataset.dataset[3]
        self.mdl.fit(self.X, self.Y)
        return GenericModel(self.mdl.export_model())


class KerasTuner(ModelTuner):
    """Keras tuner based model tuner."""

    def __init__(self: "KerasTuner", layers_range: List = [1, 6]) -> None:
        """Initialize model tuner.

        Args:
            layers_range (List): Range of hidden layers to search.
        """
        self.layers_range = layers_range
        self.input_shape = (32, 32)
        self.output_shape = 1

    def fit(self: "KerasTuner", dataset: GenericOutputDM) -> GenericModel:
        """Fit an optimal model using the given dataset.

        Args:
            dataset (GenericOutputDM): Dataset to fit an optimal model.

        Returns:
            GenericModel : Student model as wrapped in a generic model class.
        """
        self.X = dataset.dataset[2]
        self.y = dataset.dataset[3]
        self._search(self.X, self.y)
        self._get_best_epoch(self.X, self.y)
        self._final_train(self.X, self.y)
        return GenericModel(self.hypermodel)

    def _model_builder(self: "KerasTuner", hp: Any):
        model = keras.Sequential()
        hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
        model.add(
            keras.layers.Dense(
                units=hp_units, activation="relu", input_shape=(self.input_shape,)
            )
        )
        for i in range(hp.Int("layers", self.layers_range[0], self.layers_range[0])):
            model.add(
                keras.layers.Dense(
                    units=hp.Int(
                        "units_" + str(i), min_value=32, max_value=512, step=32
                    ),
                    activation="relu",
                )
            )
        model.add(keras.layers.Dense(self.output_shape))
        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss="mean_squared_error",
            metrics=None,
        )

        return model

    def _search(self: "KerasTuner", X: Any, y: Any) -> None:
        self.tuner = kt.Hyperband(
            self._model_builder,
            objective="val_loss",
            max_epochs=10,
            factor=3,
            project_name="trials",
        )
        stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
        self.tuner.search(
            X, y, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=True
        )
        self.best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]

    def _get_best_epoch(self: "KerasTuner", X: Any, y: Any) -> None:
        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        model = self.tuner.hypermodel.build(self.best_hps)
        history = model.fit(X, y, epochs=50, validation_split=0.2)

        val_per_epoch = history.history["val_loss"]
        self.best_epoch = val_per_epoch.index(min(val_per_epoch)) + 1
        print("Best epoch: %d" % (self.best_epoch,))

    def _final_train(self: "KerasTuner", X: Any, y: Any):
        self.hypermodel = self.tuner.hypermodel.build(self.best_hps)

        # Retrain the model
        self.hypermodel.fit(X, y, epochs=self.best_epoch, validation_split=0.2)
