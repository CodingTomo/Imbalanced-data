# outlier detection
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN

# other
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import pandas as pd
import utilities


def anomaly_detection(data, label):
    X = data[data.select_dtypes('number').columns.tolist()]
    y = data[label]
    y = y.values
    X = X.drop([label], axis=1)

    sc = StandardScaler()
    X = pd.DataFrame(data=sc.fit_transform(X), columns=X.columns)

    ifo = IForest(contamination=0.01, behaviour='new', n_estimators=1000, max_samples=1024, n_jobs=-1, verbose=1)
    ifo.fit(X)
    ifo_pred = ifo.labels_
    print('ROC score for Isolation forest: ', roc_auc_score(y, ifo_pred))
    utilities.plot_outlier_scores(y, ifo.decision_scores_, bw=0.1,
                                  title='Fraud, Isolation forest. (n_estimators={})'.format(ifo.n_estimators))

    ae = AutoEncoder(
        hidden_neurons=[25, 20, 15, 20, 25],
        hidden_activation='relu',
        output_activation='sigmoid',
        optimizer='adam',
        epochs=20,
        batch_size=128,
        dropout_rate=0.2,
        l2_regularizer=0.0,
        validation_size=0.1,
        preprocessing=False,
        verbose=1,
        random_state=1,
        contamination=0.01
    )
    ae.fit(X)
    ae_pred = ae.labels_
    print('ROC score for Autoencoder: ', roc_auc_score(y, ae_pred))
    utilities.plot_outlier_scores(y, ae.decision_scores_, bw=0.1,
                                  title='Fraud, Autoencoder. (epochs={})'.format(ae.epochs))

    # Too long to train, under-sample needed
    lof = LOF(n_neighbors=int(y.sum()*1.3), contamination=0.01, n_jobs=-1)
    lof.fit(X)
    lof_pred = lof.labels_
    print('ROC score for LOF: ', roc_auc_score(y, lof_pred))
    utilities.plot_outlier_scores(y, lof.decision_scores_, bw=0.1,
                                  title='Fraud, Local outliers factor. (n_neighbors={})'.format(lof.n_neighbors))

    return y, ifo_pred, ae_pred, lof_pred
