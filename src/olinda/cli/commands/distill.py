import click
import os

from . import olinda_cli
from ...distillation import Distiller

def distill_cmd():
    @olinda_cli.command(help="Distill model to ONNX surrogate model")
    @click.option("--model", "-m", type=click.STRING, help="Path to teacher model that will be distilled")
    @click.option("--output_path", "-o", type=click.STRING, required=False, default=None, help="Path and file name of onnx output file for distilled student model")
    @click.option("--test_pipeline", "-t", is_flag=True, show_default=False, default=False, help="Run the pipeline with a 10x smaller reference library")

    def distill(model, output_path, test_pipeline):
        if test_pipeline:
            d = Distiller(test=True)
        else:
            d = Distiller()

        if output_path is None:
            output_path = os.path.join(model, "distill", "distilled_model.onnx")
        student_model = d.distill(model, output_path)
