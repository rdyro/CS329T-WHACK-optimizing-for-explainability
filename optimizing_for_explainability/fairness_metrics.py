import numpy as np
from collections import namedtuple
import pandas as pd

"""
Function inspired by https://github.com/gpleiss/equalized_odds_and_calibration/blob/master/eq_odds.py
"""


class Model(namedtuple("Model", "pred label")):
    def logits(self):
        raw_logits = np.clip(np.log(self.pred / (1 - self.pred)), -100, 100)
        return raw_logits

    def num_samples(self):
        return len(self.pred)

    def base_rate(self):
        """
        Percentage of samples belonging to the positive class
        """
        return np.mean(self.label)

    def accuracy(self):
        return self.accuracies().mean()

    def precision(self):
        return (self.label[self.pred.round() == 1]).mean()

    def recall(self):
        return (self.label[self.label == 1].round()).mean()

    def tpr(self):
        """
        True positive rate
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 1))

    def fpr(self):
        """
        False positive rate
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 0))

    def tnr(self):
        """
        True negative rate
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 0))

    def fnr(self):
        """
        False negative rate
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 1))

    def fn_cost(self):
        """
        Generalized false negative cost
        """
        return 1 - self.pred[self.label == 1].mean()

    def fp_cost(self):
        """
        Generalized false positive cost
        """
        return self.pred[self.label == 0].mean()

    def accuracies(self):
        return self.pred.round() == self.label

    def __repr__(self):
        return "\n".join(
            [
                "Accuracy:\t%.3f" % self.accuracy(),
                "F.P. cost:\t%.3f" % self.fp_cost(),
                "F.N. cost:\t%.3f" % self.fn_cost(),
                "Base rate:\t%.3f" % self.base_rate(),
                "Avg. score:\t%.3f" % self.pred.mean(),
            ]
        )


def eq_odds(groups):
    print("Equalized Odds")
    vals = [group.tnr() for group in groups]
    print(
        "Groups TP rate:"
        + "\t".join([f"{group.tpr():.3f}" for group in groups])
    )
    print(
        "Groups TN rate:"
        + "\t".join([f"{val:.3f}" for val in vals])
    )
    print()
    return vals


def eq_opportunity(groups):
    print("Equality of Opportunity")
    vals = [group.tpr() for group in groups]
    print(
        "Groups TP rate:"
        + "\t".join([f"{val:.3f}" for val in vals])
    )
    print()
    return vals


def dem_parity(groups):
    print("Demographic Parity")
    vals = [group.pred.mean() for group in groups]
    print(
        "Group predicted positive rate:"
        + "\t".join([f"{val:.3f}" for val in vals])
    )
    print()
    return vals


def pred_parity(groups):
    print("Predictive Parity")
    vals = [group.precision() for group in groups]
    print("Group precision :" + "\t".join([f"{val:3f}" for val in vals]))
    print()
    return vals


def accuracy_compare(groups):
    print("Accuracy")
    vals = [group.accuracy() for group in groups]
    print("Group accuracy:" + "\t".join([f"{val:3f}" for val in vals]))
    print()
    return vals


"""
Computes desired fairness metric for a dataset with two groups.
Inputs:
metric: should be one of 'all', 'equalized_odds', 'equality_of_opportunity', 'demographic_parity', 'predictive_parity', 'accuracy'
data: should be a pandas dataframe.
pred_col: default 'prediction' (a score between 0 and 1), 
label_col: default 'label' (ground truth - either 0 or 1), 
"""


def fairness_metric(
    data,
    group_idxs,
    metric="all",
    pred_col="prediction",
    label_col="label",
):
    # data = pd.read_csv(data_filename, sep="\t")
    groups_data = [data.loc[data[group_idx] == 1] for group_idx in group_idxs]
    groups_model = [
        Model(group_data[pred_col], group_data[label_col])
        for group_data in groups_data
    ]
    if metric == "equalized_odds":
        eq_odds(groups_model)
    elif metric == "equality_of_opportunity":
        eq_opportunity(groups_model)
    elif metric == "demographic_parity":
        dem_parity(groups_model)
    elif metric == "predictive_parity":
        pred_parity(groups_model)
    elif metric == "accuracy":
        accuracy_compare(groups_model)
    elif metric == "all":
        tnr = eq_odds(groups_model)
        tpr = eq_opportunity(groups_model)
        mean = dem_parity(groups_model)
        prec = pred_parity(groups_model)
        acc = accuracy_compare(groups_model)
        return dict(tnr=tnr, tpr=tpr, mean=mean, prec=prec, acc=acc)
    else:
        raise ValueError("Metric not recognized: %s" % metric)
    return


if __name__ == "__main__":
    fairness_metric("./example.csv", metric="all")
