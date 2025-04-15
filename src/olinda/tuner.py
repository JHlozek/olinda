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

from torch.utils.data import DataLoader
from lightning import pytorch as pl
from chemprop.data.collate import collate_batch
from chemprop.nn import BondMessagePassing, NormAggregation, RegressionFFN, metrics
from chemprop import models

import ray
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import (RayDDPStrategy, RayLightningEnvironment,
                                 RayTrainReportCallback, prepare_trainer)
from ray.train.torch import TorchTrainer
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import FIFOScheduler

from olinda.data import GenericOutputDM, TensorflowDatasetWrapper, ChempropDataset
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

class ChempropTuner(ModelTuner):
    """ChemProp based model tuner."""

    def __init__(self: "ChempropTuner") -> None:
        """Initialize model tuner.

        Args:
        """
        pass

    def fit(self: "ChempropTuner", datamodule: GenericOutputDM) -> GenericModel:
        """Fit an optimal model using the given dataset.

        Args:
            datamodule (GenericOutputDM): Datamodule to fit an optimal model.

        Returns:
            GenericModel : Student model as wrapped in a generic model class.
        """
        self.datamodule = datamodule
        #self.search_datamodule = copy.deepcopy(self.datamodule)
        #train_search_dataset = 
        #val_search_dataset = 

        train_dataset = ChempropDataset(self.datamodule, "train").dataset
        val_dataset = ChempropDataset(self.datamodule, "val").dataset
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch, num_workers=4)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch, num_workers=4)
        
        self.hyperparameter_search(train_dataloader, val_dataloader)
        return self.final_train(train_dataloader, val_dataloader)

    def _build_model(self: "ChempropTuner", config: dict):
        """Build MPNN model with specified hyperparameters.

        Args:
            config (dict): Dictionary containing hyperparameters for current trial.
        """

        depth = int(config["depth"])
        ffn_hidden_dim = int(config["ffn_hidden_dim"])
        ffn_num_layers = int(config["ffn_num_layers"])
        message_hidden_dim = int(config["message_hidden_dim"])

        mp = BondMessagePassing(d_h=message_hidden_dim, depth=depth)
        agg = NormAggregation()
        ffn = RegressionFFN(input_dim=message_hidden_dim, hidden_dim=ffn_hidden_dim, n_layers=ffn_num_layers)
        metric_list = [metrics.RMSE(), metrics.MAE()]
        model = models.MPNN(mp, agg, ffn, batch_norm=True, metrics=metric_list)
        return model

    def _train_trial_model(self: "ChempropTuner", config: dict, train_dataloader: DataLoader, val_dataloader: DataLoader):
        """Train iteration of model during hyperparameter optimization.

        Args:
            config (dict): Dictionary containing hyperparameters for current trial.
            train_dataloader (DataLoader): PyTorch DataLoader for training data.
            val_dataloader (DataLoader): PyTorch DataLoader for validation data.
        """
        model = self._build_model(config)
        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=3, # number of epochs to train for
            # below are needed for Ray and Lightning integration
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
        )

        trainer = prepare_trainer(trainer)
        trainer.fit(model, train_dataloader, val_dataloader)

    def hyperparameter_search(self: "ChempropTuner", train_dataloader: DataLoader, val_dataloader: DataLoader):
        """Train iteration of model during hyperparameter optimization.

        Args:
            train_dataloader (DataLoader): PyTorch DataLoader for training data.
            val_dataloader (DataLoader): PyTorch DataLoader for validation data.
        """
        search_space = {
            "depth": tune.qrandint(lower=2, upper=6, q=1),
            "ffn_hidden_dim": tune.qrandint(lower=100, upper=1000, q=100),
            "ffn_num_layers": tune.qrandint(lower=1, upper=3, q=1),
            "message_hidden_dim": tune.qrandint(lower=100, upper=1000, q=100),
        }
        ray.init(ignore_reinit_error=True)
        scheduler = FIFOScheduler()

        # Scaling config controls the resources used by Ray
        scaling_config = ScalingConfig(
            num_workers=1,
            use_gpu=False, # change to True if you want to use GPU
        )

        # Checkpoint config controls the checkpointing behavior of Ray
        checkpoint_config = CheckpointConfig(
            num_to_keep=1, # number of checkpoints to keep
            checkpoint_score_attribute="val_loss", # Save the checkpoint based on this metric
            checkpoint_score_order="min", # Save the checkpoint with the lowest metric value
        )
        run_config = RunConfig(
            checkpoint_config=checkpoint_config,
            #storage_path=hpopt_save_dir / "ray_results", # directory to save the results
        )
        ray_trainer = TorchTrainer(
            lambda config: self._train_trial_model(config, train_dataloader, val_dataloader),
            scaling_config=scaling_config,
            run_config=run_config,
        )
        search_alg = HyperOptSearch(
            n_initial_points=1, # number of random evaluations before tree parzen estimators
            random_state_seed=42,
        )
        
        tune_config = tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=10, # number of trials to run
            scheduler=scheduler,
            search_alg=search_alg,
            trial_dirname_creator=lambda trial: str(trial.trial_id), # shorten filepaths
        )

        tuner = tune.Tuner(
            ray_trainer,
            param_space={
                "train_loop_config": search_space,
             },
            tune_config=tune_config,
        )
        self.hyper_results = tuner.fit()

    def final_train(self: "ChempropTuner", train_dataloader: DataLoader, val_dataloader: DataLoader):
        """Train optimized model.

        Args:
            train_dataloader (DataLoader): PyTorch DataLoader for training data.
            val_dataloader (DataLoader): PyTorch DataLoader for validation data.
        """
        best_result = self.hyper_results.get_best_result()
        best_config = best_result.config
        print("Best hyperparameters:")
        print(best_config['train_loop_config'])
        
        if best_config is not None:
            self.final_model = self._build_model(best_config['train_loop_config'])
        else:
            mp = BondMessagePassing()
            agg = NormAggregation()
            ffn = RegressionFFN()
            metric_list = [metrics.RMSE(), metrics.MAE()]
            self.final_model = models.MPNN(mp, agg, ffn, batch_norm=True, metrics=metric_list)

        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,  # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
            enable_progress_bar=True,
            accelerator="auto",
            devices=1,
            max_epochs=30,  # number of epochs to train for
        )
        trainer.fit(self.final_model, train_dataloader, val_dataloader)
        return GenericModel(self.final_model)

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
