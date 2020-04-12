# system modules
import os
import sys
from sklearn.utils import shuffle

# custom modules
import config
import data_exploration
import utilities
import supervised
import unsupervised


# workspace
print("Actual workspace: ", os.getcwd())

# logger
# sys.stdout = config.Logger("log/log.txt")

# loading data and params and elementary exploration
loader = config.Loader()
params = config.Params()

# elementary exploration
data_exploration.summary(loader)

# manage missing
starting_data = data_exploration.manage_missing(loader.data)

# variable distribution per class and linear correlation
data_exploration.variable_distribution(starting_data)
data_exploration.linear_correlation(starting_data)

# shuffle
starting_data = shuffle(starting_data).reset_index(drop=True)

# Supervised - naive attempt
supervised.naive_attempt(starting_data, params.label)

# Supervised - re-balancing attempts - grid search
result_rf, result_xgb = supervised.grid_search(starting_data, params.label)
best_model = utilities.manage_supervised_result(result_rf, result_xgb, params.decision_metric)
best_model.reset_index(inplace=True, drop=True)

# Supervised - re-balancing attempts - train best model
model_metric, model_class_metric, cm = supervised.best_model(starting_data,
                                                             best_model,
                                                             params.label,
                                                             params.decision_threshold
                                                             )

# Unsupervised - sample to speed-up training
loader.shuffle_sample()
starting_data = data_exploration.manage_missing(loader.data)
starting_data = shuffle(starting_data).reset_index(drop=True)

# Unsupervised - anomaly detection
y, ifo_pred, ae_pred, lof_pred = unsupervised.anomaly_detection(starting_data, params.label)
result_unsupervised_methods = utilities.manage_unsupervised_result(y, ifo_pred, ae_pred, lof_pred)

