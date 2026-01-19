import os
import pandas as pd
import stylia
from stylia import TWO_COLUMNS_WIDTH

INDIVIDUAL_FIGSIZE = (TWO_COLUMNS_WIDTH / 2, TWO_COLUMNS_WIDTH / 2)

class BasePlot(object):
    def __init__(self, ax, path, figsize=None):
        self.path=path
        if ax is None:
            if figsize is None:
                figsize = INDIVIDUAL_FIGSIZE
            _, ax = stylia.create_figure(1, 1, width=figsize[0], height=figsize[1])
        self.ax = ax[0]

    def save(self, name):
        stylia.save_figure(
            os.path.join(self.path, name + ".png")
        )