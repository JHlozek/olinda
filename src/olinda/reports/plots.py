import numpy as np

from sklearn import metrics
from sklearn.metrics import auc, roc_curve, r2_score, mean_absolute_error

import matplotlib as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd

from stylia import NamedColors, NamedColorMaps, ContinuousColorMap
from olinda.reports.base import BasePlot

named_colors = NamedColors()
named_cmaps = NamedColorMaps()

class RocCurvePlot(BasePlot):
    def __init__(self, bin_true, zaira_pred, olinda_pred, ax, path):
        BasePlot.__init__(self, ax=ax, path=path, figsize=(3, 3))
        self.name = "roc-curve"
        ax = self.ax
        cmap = ContinuousColorMap(cmap="spectral")
        cmap.fit([0, 1])

        fpr_z, tpr_z, _ = roc_curve(bin_true, zaira_pred)
        auroc_z = auc(fpr_z, tpr_z)
        color = named_colors.black
        ax.plot(fpr_z, tpr_z, color=color, zorder=10000, lw=1, label="ZairaChem")
        
        fpr_o, tpr_o, _ = roc_curve(bin_true, olinda_pred)
        auroc_o = auc(fpr_o, tpr_o)
        color = cmap.transform([auroc_o])[0]
        ax.plot(fpr_o, tpr_o, color=color, zorder=10001, lw=1, label="Olinda")
        ax.fill_between(fpr_o, tpr_o, color=color, alpha=0.5, lw=0, zorder=1000)

        ax.plot([0, 1], [0, 1], color=named_colors.gray, lw=1)
        ax.set_title("AUROC = {0} | AUROC maintained = {1}".format(round(auroc_o, 2), round(auroc_o / auroc_z, 2)))
        ax.set_xlabel("1-Specificity (FPR)")
        ax.set_ylabel("Sensitivity (TPR)")
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.legend(loc="lower right").set_zorder(10002)

class RegressionPlotRaw(BasePlot):
    def __init__(self, zaira_pred, olinda_pred, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        self.name = "regression-raw"
        ax = self.ax
        ax.scatter(zaira_pred, olinda_pred, color=named_colors.blue, s=15, alpha=0.7)
        ax.set_xlabel("ZairaChem Prediction")
        ax.set_ylabel("Olinda Prediction")
        ax.set_title(
            "R2 = {0} | MAE = {1}".format(
                round(r2_score(zaira_pred, olinda_pred), 3), round(mean_absolute_error(zaira_pred, olinda_pred), 3)
            )
        )
