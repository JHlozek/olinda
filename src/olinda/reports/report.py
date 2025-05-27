import os
import pandas as pd
import onnx_runner
from olinda.reports.plots import RocCurvePlot, RegressionPlotRaw

class Reporter:
    def __init__(self, zaira_path, olinda_path):
        self.zaira_path = zaira_path
        self.olinda_path = olinda_path
        
        zaira_preds_df = pd.read_csv(os.path.join(self.zaira_path, "distill", "original_training_set.csv"))
        zaira_true_df = pd.read_csv(os.path.join(self.zaira_path, "report", "output_table.csv"))
        
        self.zaira_train_preds = zaira_preds_df["prediction"].tolist()
        self.zaira_train_true = zaira_true_df["true-value"].tolist()

        onnx_model = onnx_runner.ONNX_Runner(olinda_path)
        self.olinda_preds = onnx_model.predict(zaira_preds_df["smiles"].tolist())

    def report(self):
        roc = RocCurvePlot(self.zaira_train_true, self.zaira_train_preds, self.olinda_preds, ax=None, path=os.path.join(self.zaira_path, "distill"))
        roc.save("olinda_roc_curve")

        reg = RegressionPlotRaw(self.zaira_train_preds, self.olinda_preds, ax=None, path=os.path.join(self.zaira_path, "distill"))
        reg.save("olinda_zaira_reg")
