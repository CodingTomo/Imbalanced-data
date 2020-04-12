# plot library
import matplotlib.pyplot as plt
import seaborn as sns


def summary(dataset):
    X = dataset.data.drop(['Class'], axis=1)
    y = dataset.data['Class']
    print(" Train null = {} |  Label null = {}"
          .format(X.isnull().sum().sum(),
                  y.isnull().sum()))
    print("Total rows number {}. \n Unique values in the train set: \n {}"
          .format(len(X), X.nunique()))
    print("Label distribution: \n {} \n In percentage \n {}"
          .format(y.value_counts(), y.value_counts()/len(y)))


def manage_missing(data):
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    print("Number of missing: ", data.isnull().sum().sum())
    print("Row count without duplicates: ", len(data))
    return data


def variable_distribution(data):
    for var in data.columns:
        data.groupby("Class")[var].hist(bins=25, figsize=(5, 5), density=True, alpha=0.4)
        plt.savefig("plots/var_dist_"+str(var)+".png")
        plt.clf()


def linear_correlation(data):
    corrMatrix = data.corr().round(2)
    plt.figure(figsize=(20, 20))
    sns.heatmap(corrMatrix, annot=True)
    plt.savefig("plots/covariance_matrix.png")
    plt.clf()
    print('Linear dependence between 0/1 labels and the other variables:\n ', corrMatrix['Class'])












