"""

checking the history csv files for each run

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


class AnalyseData(object):

    def __init__(self):

        pass

    def csv_analysis(self, filepath, metric, str_metric):

        table = pd.read_csv(filepath)

        sb.lineplot(x=table["epoch"], y=table[metric], color="blue", linewidth=2)
        p = sb.lineplot(x=table["epoch"], y=table["val_" + metric], color="orange", linewidth=2)

        p.set_ylabel(str_metric, fontsize=16)
        p.set_xlabel("Epoch", fontsize=16)

        p.tick_params(axis="x", labelsize=16)
        p.tick_params(axis="y", labelsize=16)

        plt.legend(('Training', 'Validation'), prop={"size": 12})

        plt.show()

