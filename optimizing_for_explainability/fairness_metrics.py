import numpy as np
from collections import namedtuple
import pandas as pd 

"""
Function inspired by https://github.com/gpleiss/equalized_odds_and_calibration/blob/master/eq_odds.py
"""
class Model(namedtuple('Model', 'pred label')):
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
        return '\n'.join([
            'Accuracy:\t%.3f' % self.accuracy(),
            'F.P. cost:\t%.3f' % self.fp_cost(),
            'F.N. cost:\t%.3f' % self.fn_cost(),
            'Base rate:\t%.3f' % self.base_rate(),
            'Avg. score:\t%.3f' % self.pred.mean(),
        ])

def eq_odds(group_0, group_1):
    print('Equalized Odds')
    print('Group 0 / Group 1 TP rate:\t%.3f \t / \t %.3f' % (group_0.tpr(), group_1.tpr()))
    print('Group 0 / Group 1 TN rate:\t%.3f \t / \t %.3f' % (group_0.tnr(), group_1.tnr()))
    print()

def eq_opportunity(group_0, group_1):
    print('Equality of Opportunity')
    print('Group 0 / Group 1 TP rate:\t%.3f \t / \t %.3f' % (group_0.tpr(), group_1.tpr()))
    print()

def dem_parity(group_0, group_1):
    print('Demographic Parity')
    print('Group 0 / Group 1 predicted positive rate:\t%.3f \t / \t %.3f' % (group_0.pred.mean(), group_1.pred.mean()))
    print()

def pred_parity(group_0, group_1):
    print('Predictive Parity')
    print('Group 0 / Group 1 precision:\t%.3f \t / \t %.3f' % (group_0.precision(), group_1.precision()))
    print()

def accuracy_compare(group_0, group_1):
    print('Accuracy')
    print('Group 0 / Group 1 accuracy:\t%.3f \t / \t %.3f' % (group_0.accuracy(), group_1.accuracy()))
    print()

"""
Computes desired fairness metric for a dataset with two groups.
Inputs:
metric: should be one of 'all', 'equalized_odds', 'equality_of_opportunity', 'demographic_parity', 'predictive_parity', 'accuracy'
data_filename: should be a csv file that contains the following columns,
pred_col: default 'prediction' (a score between 0 and 1), 
label_col: default 'label' (ground truth - either 0 or 1), 
group_col: default 'group' (group assignment - either 0 or 1)
"""
def fairness_metric(data_filename, metric = 'all', pred_col = 'prediction', label_col = 'label', group_col = 'group'):
    data = pd.read_csv(data_filename, sep='\t')
    group_0_data = data[data[group_col]==0]
    group_1_data = data[data[group_col]==1]
    group_0_model = Model(group_0_data[pred_col], group_0_data[label_col])
    group_1_model = Model(group_1_data[pred_col], group_1_data[label_col])
    if metric == 'equalized_odds':
        eq_odds(group_0_model, group_1_model)
    elif metric == 'equality_of_opportunity':
        eq_opportunity(group_0_model, group_1_model)
    elif metric == 'demographic_parity':
        dem_parity(group_0_model, group_1_model)
    elif metric == 'predictive_parity':
        pred_parity(group_0_model, group_1_model)
    elif metric == 'accuracy':
        accuracy_compare(group_0_model, group_1_model)
    elif metric == 'all':
        accuracy_compare(group_0_model, group_1_model)
        dem_parity(group_0_model, group_1_model)
        pred_parity(group_0_model, group_1_model)
        eq_opportunity(group_0_model, group_1_model)
        eq_odds(group_0_model, group_1_model)
    else:
        raise ValueError('Metric not recognized: %s' % metric)
    return

if __name__ == '__main__':
    fairness_metric('./example.csv', metric='all')