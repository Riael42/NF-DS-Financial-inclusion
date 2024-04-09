#### IMPORTS

# dataframe and plotting
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb

#### IMPORTS


###########MODEL

def project_model(training_data, random_seed):

    ##FIX
    training_data = pd.read_csv(training_data)
    ##FIX

    #print(training_data.info())
    #print(training_data.shape)
    #print(training_data.head())

    #todo: maybe split everything so we can do niter_forest = 3 num_cv_xgboost=21, somethin like that
    stratify = 0 #switch whether or not to stratify target_variable
    target_variable = "bank_account" #maybe we use this for the next project too
    niter = 100 #number iterators
    num_cv = 5 #num cross validation folds
    verbose=0 #quiet 0 1 2 loud
    num_jobs = -1 #-1 all cpu cores, 1 disables parallelization, 2 uses specified number
    scoring = "f1" #if we want to change scoring for whatever reason
    #todo: random_train_split make it so you randomize splits between models

    #todo randomize the random parameters for the random models
    #not random parameters for logistic
    penaltea = ['l2'] #penalty for logistic regression
    C_log = [0.001, 0.01, 0.1, 1, 10, 100] #C for logistic regression
    slogver = [ 'lbfgs', 'liblinear'] #solver for logistic regression
    miter = [100, 150, 200] #max iterations for logistic regression

    #not random parameters for random forest
    nestimators = [10, 100, 1000] # num estimators for random forest
    big_deep = [10, 20, 30] #maximum depth for random forest
    min_forest_split = [2, 4, 6] # min_samples split
    min_tea_leafs = [1, 2, 4] #min_samples_leafs
    bootstrap = [True, False] # boostrap
    cryterio = ['gini', 'entropy'] # criterion

    #not random params for random xgboost
    xbgeta = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5] #eta
    XgammaBOOST = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  #gamma 
    deep_boost = [3, 5, 7, 10] #max_depth
    min_boost_weight = [1, 3, 5, 7] #min_child_weight
    xgboosamplet = [0.6, 0.8, 1.0] #subsample
    xgbytree = [0.6, 0.8, 1.0] #colsample_bytree
    xgboojective = ["binary:logistic"] #objective
    xgauc = ["auc"] #eval_metric
    xgscale = [1, 1] #scale_pos_weight

    #voting classifier
    voting = "hard" #can be soft


    #data split
    target = training_data[target_variable]
    train = training_data.drop(target_variable, axis=1)
    target.fillna(0, inplace=True)
    train.fillna(0, inplace=True)
    #print("shape: ", target.shape)
    #print("unique values: ", target.unique())
    #print(train.dtypes)
    #print(target.dtypes)
    #print(target)
    if (stratify == 1):
        X, X_train, y, y_train = train_test_split(train, target, test_size=0.2, random_state=rs_one, stratify=target_variable)
    else:
        X, X_train, y, y_train = train_test_split(train, target, test_size=0.2, random_state=rs_one)
    #Darth Loger
    #train logistic
    tuned_logistic = LogisticRegression()

    params = {
        'penalty': penaltea, 
        'C': C_log,
        'solver': slogver,
        'max_iter': miter
    }

    tuned_logistic = RandomizedSearchCV(estimator=tuned_logistic, 
                                        param_distributions=params, 
                                        n_iter=niter, cv=num_cv, 
                                        random_state=random_seed, verbose=verbose,
                                        n_jobs = num_jobs,
                                        scoring=scoring)
    tuned_logistic.fit(X_train, y_train)
    tuned_logistic = tuned_logistic.best_estimator_

    #Darth Lorax
    #train random forest
    tuned_forest = RandomForestClassifier()

    params = {
    'n_estimators': nestimators, 
    'max_depth': big_deep,
    'min_samples_split': min_forest_split,
    'min_samples_leaf': min_tea_leafs,
    'bootstrap': bootstrap,
    'criterion': cryterio
    }

    tuned_forest = RandomizedSearchCV(estimator=tuned_forest,
                                      param_distributions=params,
                                      n_iter=niter, cv=num_cv,
                                      random_state=random_seed, verbose=verbose,
                                      n_jobs = num_jobs,
                                      scoring=scoring)
    tuned_forest.fit(X_train, y_train)
    tuned_forest = tuned_forest.best_estimator_

    #Darth XGBious
    #train xgboost
    tuned_xgboost = xgb.XGBClassifier()

    params = {
        "eta" : xbgeta,
        "gamma": XgammaBOOST,
        "max_depth" : deep_boost,
        "min_child_weight" : min_boost_weight,
        "subsample" : xgboosamplet,
        "colsample_bytree" : xgbytree,
        "objective" : xgboojective,
        "eval_metric" : xgauc,
        "scale_pos_weight" : xgscale
    }
    tuned_xgboost = RandomizedSearchCV(estimator=tuned_xgboost,
                                       param_distributions=params,
                                       n_iter=niter, cv=num_cv,
                                       random_state=random_seed, verbose=verbose,
                                       n_jobs = num_jobs,
                                       scoring=scoring)
    tuned_xgboost.fit(X_train, y_train)
    tuned_xgboost = tuned_xgboost.best_estimator_

    sith_council = VotingClassifier(estimators = [
                                    ("logistic", tuned_logistic),
                                    ("random_forest", tuned_forest),
                                    ("xgboost", tuned_xgboost)], voting=voting
    )
    sith_council.fit(X_train, y_train)

    #The great wall of prints

    print("----------")
    print("Logistic Regression")
    print("----------")
    print("Accuracy on train data:", accuracy_score(y_train, tuned_logistic.predict(X_train)))
    print("Classification report on train data:")
    print(classification_report(y_train, tuned_logistic.predict(X_train)))
    print("----------")
    print("Accuracy on test data:", accuracy_score(y, tuned_logistic.predict(X)))
    print("Classification report on test data:")
    print(classification_report(y, tuned_logistic.predict(X)))
    print("----------")
    print("Random Forest")
    print("----------")
    print("Accuracy on train data:", accuracy_score(y_train, tuned_forest.predict(X_train)))
    print("Classification report on train data:")
    print(classification_report(y_train, tuned_forest.predict(X_train)))
    print("----------")
    print("Accuracy on test data:", accuracy_score(y, tuned_forest.predict(X)))
    print("Classification report on test data:")
    print(classification_report(y, tuned_forest.predict(X)))
    print("----------")
    print("XGBoost")
    print("----------")
    print("Accuracy on train data:", accuracy_score(y_train, tuned_xgboost.predict(X_train)))
    print("Classification report on train data:")
    print(classification_report(y_train, tuned_xgboost.predict(X_train)))
    print("----------")
    print("Accuracy on test data:", accuracy_score(y, tuned_xgboost.predict(X)))
    print("Classification report on test data:")
    print(classification_report(y, tuned_xgboost.predict(X)))
    print("----------")
    print("Voting Classifier")
    print("----------")
    print("Accuracy on train data:", accuracy_score(y_train, sith_council.predict(X_train)))
    print("Classification report on train data:")
    print(classification_report(y_train, sith_council.predict(X_train)))
    print("----------")
    print("Accuracy on test data:", accuracy_score(y, sith_council.predict(X)))
    print("Classification report on test data:")
    print(classification_report(y, sith_council.predict(X)))
    print("----------")


###########MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression Tester")
    parser.add_argument("--training_data", type=str, required=True,
                        help="Give me data, or give me death! -some human")
    parser.add_argument("--random_seed", type=int, required=True, help="Random seed for splitting the data")
    args = parser.parse_args()
    print("Arguments:", args)
    project_model(training_data=args.training_data, random_seed=args.random_seed)



###########MAIN
