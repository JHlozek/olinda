"""Distillation ONNX model."""

from abc import ABC
from typing import Any

import os
import onnx
from olinda.models.base import DistillBaseModel


class DistilledModel(DistillBaseModel):
    """Distillation model in ONNX format."""
    def __init__(self, onnx_model):
        self.model = onnx_model

    def save(self: "DistilledModel", path: str) -> None:
        onnx.save(self.model, os.path.join(path))

    def load(self: "DistilledModel", path: str) -> None:
            self.model = onnx.load(os.path.join(path))
