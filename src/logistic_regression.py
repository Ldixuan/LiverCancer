from sklearn.linear_model import LogisticRegression
from sklearn import decomposition
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

import numpy as np
from matplotlib import pyplot as plt

RESOURCE_PATH = "../resource/HCCDB6"
EXP_PATH = f"{RESOURCE_PATH}/filted_gene.csv"
SURVIVAL_INFO_PATH = f"{RESOURCE_PATH}/GSE14520_Extra_Supplement.xls"


if __name__ == '__main__':
    exp_df = pd.read_csv(EXP_PATH, header=0, sep="\t")
    survival_info_df = pd.read_csv(SURVIVAL_INFO_PATH, header=0, sep="\t")

    survival_info_df_index = survival_info_df['Affy_GSM']
    survival_info_df.drop('Affy_GSM', axis=1, inplace=True)
    survival_info_df.index = survival_info_df_index

    positive_sample = []
    negative_sample = []
    X = []
    y = []
    df_index = []

    for index, row in exp_df.iteritems():
        if index == "ID":
            continue

        id = index.split("=")[0]
        if id not in survival_info_df.index:
            continue

        futime = survival_info_df.loc[id]['Survival months'] / 12
        fustat = survival_info_df.loc[id]['Survival status']

        if fustat == 1 and futime < 3:
            y.append(-1)
        elif futime >= 3:
            y.append(1)
        else:
            continue

        X.append(row)

    std_slc = StandardScaler()
    pca = decomposition.PCA()
    logistic_Reg = LogisticRegression(penalty='l2', max_iter=10000)

    pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('pca', pca),
                           ('logistic_Reg', logistic_Reg)])

    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
    param_grid = {"logistic_Reg__C": np.logspace(-3, 3, 7)}

    # lr = LogisticRegression(penalty="l2", max_iter=10000)
    clf = GridSearchCV(pipe, param_grid, cv=StratifiedKFold(10),
                       scoring=scoring, refit='AUC', return_train_score=True, verbose=2)
    clf.fit(X, y)

    print(clf.best_score_)
    print(clf.best_estimator_)
    print(clf.best_params_)
