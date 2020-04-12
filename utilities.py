# utils
import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# metrics
from sklearn.metrics import roc_auc_score, average_precision_score


def manage_supervised_result(result_rf, result_xgb, metric):
    result_rf['classifier'] = 'rf'
    result_xgb['classifier'] = 'xgb'

    result_rf = result_rf.sort_values(by=[metric], ascending=False).head(1)
    result_xgb = result_xgb.sort_values(by=[metric], ascending=False).head(1)

    best_model_per_type = pd.concat([result_rf[['classifier', metric, 'params']],
                                     result_xgb[['classifier', metric, 'params']]
                                     ])

    best_model = best_model_per_type.sort_values(by=[metric], ascending=False).head(1)

    return best_model


def manage_unsupervised_result(y, ifo_pred, ae_pred, lof_pred):
    y = pd.Series(y, name='true', dtype='int32')
    ifo = pd.Series(ifo_pred, name='ifo', dtype='int32')
    ae = pd.Series(ae_pred, name='ae', dtype='int32')
    lof = pd.Series(lof_pred, name='lof')

    summary_table = pd.concat([y, ifo, ae, lof], axis=1)

    summary_table['detected'] = np.where((summary_table['ifo'] == 1) |
                                         (summary_table['ae'] == 1) |
                                         (summary_table['lof'] == 1),
                                         1, 0)

    summary_table['true positive'] = np.where((summary_table['true'] == 1) & (summary_table['detected'] == 1), 1, 0)

    summary_table['false negative'] = np.where((summary_table['true'] == 1) & (summary_table['detected'] == 0), 1, 0)

    summary_table['false positive'] = np.where((summary_table['true'] == 0) & (summary_table['detected'] == 1), 1, 0)

    summary_table['true negative'] = np.where((summary_table['true'] == 0) & (summary_table['detected'] == 0), 1, 0)

    print('Checking {} case, the coverage on frauds is about {:.0%}'
          .format(summary_table['detected'].sum(),
                  summary_table['true positive'].sum()/summary_table['true'].sum()))

    return summary_table


def plot_outlier_scores(y_true, scores, title='', **kdeplot_options):
    roc_score = roc_auc_score(y_true, scores)
    pr_score = average_precision_score(y_true, scores)

    classify_results = pd.DataFrame(data=pd.concat((pd.Series(y_true), pd.Series(scores)), axis=1))
    classify_results.rename(columns={0: 'true', 1: 'score'}, inplace=True)

    sns.kdeplot(classify_results.loc[classify_results.true == 0, 'score'], label='negatives',
                shade=True, **kdeplot_options)
    sns.kdeplot(classify_results.loc[classify_results.true == 1, 'score'], label='positives',
                shade=True, **kdeplot_options)
    plt.title('{} AUC-ROC: {:.3f}, AUC-PR: {:.3f}'.format(title, roc_score, pr_score))
    plt.xlabel('Predicted outlier score')
    plt.savefig("plots/"+str(title)+"feature_importance.png")
    plt.clf()

    return



