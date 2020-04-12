# manage imbalance
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# model
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, make_scorer
from sklearn.metrics import confusion_matrix, brier_score_loss
from sklearn.preprocessing import StandardScaler

# visualization
import scikitplot as skplt
import matplotlib.pyplot as plt
from xgboost import plot_importance

# other
import pandas as pd


def naive_attempt(data, label):
    X = data[data.select_dtypes('number').columns.tolist()]
    y = data[label]
    X.drop([label], axis=1, inplace=True)

    sc = StandardScaler()
    X = pd.DataFrame(data=sc.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    lc = linear_model.SGDClassifier(random_state=123, max_iter=2000, tol=1e-3, n_jobs=-1)
    lr = linear_model.LogisticRegression(random_state=123, n_jobs=-1, max_iter=2000)

    lc.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    lc_pred = lc.predict(X_test)
    lr_pred = lr.predict(X_test)

    lc_cm = confusion_matrix(y_test, lc_pred)
    lr_cm = confusion_matrix(y_test, lr_pred)

    print("Linear classifier ROC-AUC score: {}".format(roc_auc_score(y_test, lc_pred)))
    print("Logistic regression ROC-AUC score: {}".format(roc_auc_score(y_test, lr_pred)))

    return lc_cm, lr_cm


def grid_search(data, label):
    X = data[data.select_dtypes('number').columns.tolist()]
    y = data[label]
    X.drop([label], axis=1, inplace=True)

    rf = RandomForestClassifier(random_state=123)
    xgb = XGBClassifier(random_state=123, objective='binary:logistic')

    params_rf = {'rf__n_estimators': [300],
                 'rf__min_samples_leaf': [5],
                 'over__k_neighbors': [3],
                 'rf__criterion': ['gini'],
                 'rf__max_features': ['auto']
                 }

    params_xgb = {'xgb__n_estimators': [1000],
                  'xgb__learning_rate': [0.01],
                  'xgb__max_depth': [1, 2],
                  'over__k_neighbors': [3],
                  'xgb__subsample': [1],
                  'xgb__colsample_bytree': [1],
                  'xgb__gamma': [0]
                  }

    scorers = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score),
        'roc_auc_score': make_scorer(roc_auc_score),
        'f1_score': make_scorer(f1_score)
    }

    over_sampling = SMOTE(sampling_strategy=0.2)
    under_sampling = RandomUnderSampler(sampling_strategy=0.9)

    pipeline_rf = Pipeline(steps=[('over', over_sampling),
                                  ('under', under_sampling),
                                  ('rf', rf)
                                  ])

    pipeline_xgb = Pipeline(steps=[('over', over_sampling),
                                   ('under', under_sampling),
                                   ('xgb', xgb)
                                   ])

    grid_rf = GridSearchCV(pipeline_rf, params_rf, scoring=scorers, cv=2, verbose=1, n_jobs=-1, refit='roc_auc_score')
    grid_xgb = GridSearchCV(pipeline_xgb, params_xgb, scoring=scorers, cv=2, verbose=1, n_jobs=-1, refit='roc_auc_score')

    grid_rf.fit(X, y)
    grid_xgb.fit(X, y)

    print('ROC score for Random Forest: ', grid_rf.best_score_)
    print('ROC score for XGBoost: ', grid_xgb.best_score_)

    result_rf = pd.DataFrame(grid_rf.cv_results_)
    result_xgb = pd.DataFrame(grid_xgb.cv_results_)

    return result_rf, result_xgb


def best_model(data, best, label, threshold):
    X = data[data.select_dtypes('number').columns.tolist()]
    y = data[label]
    X.drop([label], axis=1, inplace=True)

    params = best['params'].loc[0]

    if best['classifier'].loc[0] == 'rf':
        model = RandomForestClassifier(n_jobs=-1,
                                       random_state=123,
                                       n_estimators=params['rf__n_estimators'],
                                       min_samples_leaf=params['rf__min_samples_leaf'],
                                       criterion=params['rf__criterion'],
                                       max_features=params['rf__max_features']
                                       )
    elif best['classifier'].loc[0] == 'xgb':
        model = XGBClassifier(n_jobs=-1,
                              random_state=123,
                              objective='binary:logistic',
                              n_estimators=params['xgb__n_estimators'],
                              learning_rate=params['xgb__learning_rate'],
                              max_depth=params['xgb__max_depth'],
                              subsample=params['xgb__subsample'],
                              colsample_bytree=params['xgb__colsample_bytree'],
                              gamma=params['xgb__gamma']
                              )
        print(model)
    else:
        print('Model not found!')

    over_sampling = SMOTE(sampling_strategy=0.2,
                          k_neighbors=params['over__k_neighbors'])
    under_sampling = RandomUnderSampler(sampling_strategy=0.9)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    steps = [('over', over_sampling), ('under', under_sampling)]
    pipeline = Pipeline(steps=steps)
    X_train, y_train = pipeline.fit_resample(X_train, y_train)

    model.fit(X_train, y_train)

    y_pred_pr_all = model.predict_proba(X_test)
    y_pred_pr = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_pr > threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    brier_test = pd.Series(brier_score_loss(y_test, y_pred_pr))
    auc_score_test = pd.Series(roc_auc_score(y_test, y_pred))
    acc_test = pd.Series(accuracy_score(y_test, y_pred))
    recall_test = pd.Series(recall_score(y_test, y_pred, average=None))
    precision_test = pd.Series(precision_score(y_test, y_pred, average=None))
    f1_test = pd.Series(f1_score(y_test, y_pred, average=None))

    model_metric = (pd.concat([brier_test, auc_score_test, acc_test], axis=1)).rename(
        columns={0: 'Brier Score', 1: 'ROC-AUC', 2: 'Accuracy'},
        index={0: "Value"})

    model_class_metric = (pd.concat([recall_test, precision_test, f1_test], axis=1)).rename(
        columns={0: 'Recall', 1: 'Precision', 2: 'F1'},
        index={0: "0 - Class", 1: "1 - Class"})

    skplt.metrics.plot_roc(y_test, y_pred_pr_all, plot_micro=False, plot_macro=False)
    plt.savefig("plots/best_supervised_model_roc_curve.png")
    plt.clf()

    if best['classifier'].loc[0] == 'rf':
        feat_importance = pd.Series(model.feature_importances_, index=X.columns)
        feat_importance.nlargest(10).plot(kind='hist')
        plt.savefig("plots/feature_importance.png")
        plt.clf()

    elif best['classifier'].loc[0] == 'xgb':
        plot_importance(model)
        plt.savefig("plots/feature_importance.png")
        plt.clf()

    else:
        print('Model not found!')

    print('fine')

    return model_metric, model_class_metric, cm
