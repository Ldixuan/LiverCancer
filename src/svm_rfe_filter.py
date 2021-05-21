from sklearn.feature_selection import RFECV
from sklearn.svm import OneClassSVM
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd

RESOURCE_PATH = "../resource/HCCDB6"
EXP_PATH = f"{RESOURCE_PATH}/diffGeneExp.txt"
SURVIVAL_INFO_PATH = f"{RESOURCE_PATH}/GSE14520_Extra_Supplement.xls"

if __name__ == '__main__':
    exp_df = pd.read_csv(EXP_PATH, header=0, sep="\t")
    survival_info_df = pd.read_csv(SURVIVAL_INFO_PATH, header=0, sep="\t")

    survival_info_df.index = survival_info_df['Affy_GSM']
    survival_info_df.drop('Affy_GSM', axis=1,inplace=True)

    # colums = survival_info_df.iloc[0]
    # survival_info_df.index = survival_info_df['PATIENT_ID']
    # survival_info_df.drop('PATIENT', axis=0, inplace=True)
    # survival_info_df.columns = colums
    # survival_info_df = survival_info_df.T
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

        futime = float(survival_info_df.loc[id]['Survival months'])

        if futime != futime:
            continue

        fustat = survival_info_df.loc[id]['Survival status']
        if fustat != fustat:
            continue

        futime = futime / 12

        if fustat == 1 and futime < 3:
            y.append(1)
        elif futime >= 3:
            y.append(0)
        else:
            continue

        X.append(row)

    param_grid = {'estimator__C': [0.0001, 0.0005, 0.001, 0.002, 0.005]}

    svc = LinearSVC(max_iter=1000000, penalty="l2")
    rfecv = RFECV(estimator=svc, scoring='accuracy', cv=StratifiedKFold(10))
    clf = GridSearchCV(rfecv, param_grid, scoring='accuracy', verbose=2)
    clf.fit(X, y)

    ranked_df = pd.DataFrame(clf.best_estimator_.ranking_, index=exp_df['ID'], columns=['Rank']).sort_values(by='Rank',
                                                                                                             ascending=True)

    print("best estimator")
    print(clf.best_estimator_.estimator_)
    print()
    print("grid scores")
    print(len(clf.best_estimator_.grid_scores_))
    print()
    print("selected gene")
    print(ranked_df[ranked_df['Rank'] == 1])
    print("number selected : " + str(len(ranked_df[ranked_df['Rank'] == 1])))
    print()
    filted_gene = ranked_df[ranked_df['Rank'] == 1].index.tolist()
    output_df = exp_df[exp_df['ID'].isin(filted_gene)]
    output_df.reset_index(drop=True, inplace=True)
    output_df.to_csv(f"{RESOURCE_PATH}/filted_gene.csv", header=True, sep='\t')

