def undersample_balance_dataset(df, minority_ratio=0.5, target_variable='bank_account'):
    """
    Undersamples the majority class in a DataFrame to balance the dataset.

    Parameters:
    - df: DataFrame containing the dataset.
    - minority_ratio: Ratio of the minority class in the balanced dataset. Default is 0.5.
    - target_variable: Name of the target variable. Default is 'bank_account'.

    Returns:
    - balanced_df: DataFrame containing the balanced dataset.
    """
    
  # Count the number of samples in each class
    class_counts = df[target_variable].value_counts(normalize = True)
    print(class_counts)

    # Determine the minority and majority classes
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()

    # Separate the dataframe into minority and majority classes
    minority_df = df[df[target_variable] == minority_class]
    majority_df = df[df[target_variable] == majority_class]

    majority_size = int(minority_ratio*minority_df.shape[0]/(1-minority_ratio))
    # Sample from majority class to match minority class ratio
    majority_sampled = majority_df.sample(majority_size)
    
    # Concatenate minority and sampled majority class
    balanced_df = pd.concat([minority_df, majority_sampled])
    class_counts = balanced_df[target_variable].value_counts()
    print(class_counts)
    
    # Shuffle the balanced dataframe
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
    return balanced_df

###########MAIN





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression Tester")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Give me data, or give me death! -some human")
    parser.add_argument("--random_seed", type=int, required=True, help="Random seed for splitting the data")
    args = parser.parse_args()
    print("Arguments:", args)
    project_model(data_path=args.data_path, random_seed=args.random_seed)





###########MAIN

#### IMPORTS

# dataframe and plotting
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemple import VOtingClassifier
from xboost import XGBClassifier

#### IMPORTS
def project_model(training_data, random_seed):

    #todo: maybe split everything so we can do niter_forest = 3 num_cv_xgboost=21, somethin like that
    stratify = 0 #switch whether or not to stratify target_variable
    random_seed = 42 #random seed for everything random
    target_variable = "bank_account" #maybe we use this for the next project too
    niter = 10 #number iterators
    num_cv = 5 #num cross validation folds
    verbose=0 #quiet 0 1 2 loud
    num_jobs = -1 #-1 all cpu cores, 1 disables parallelization, 2 uses specified number
    scoring = "f1" #if we want to change scoring for whatever reason
    #todo: random_train_split make it so you randomize splits between models


    #todo randomize the random parameters for the random models
    #not random parameters for logistic
    penaltea = ['l1', 'l2', 'elasticnet'] #penalty for logistic regression
    C_log = [0.001, 0.01, 0.1, 1, 10, 100, 1000] #C for logistic regression
    slogver = ['saga', 'lbfgs', 'liblinear', 'newton-cg'] #solver for logistic regression
    miter = [100, 1000, 10000] #max iterations for logistic regression

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
    xgboojective = "binary:logistic" #objective
    xgauc = "auc" #eval_metric
    xgscale = [1, 1] #scale_pos_weight


    #data split
    y = training_data[target_variable]
    X = training_data.drop(target_variable, axis=1)
    X, X_train, y, y_train = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=stratify)

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

    #train xgboost
    tuned_xgboost = XGBClassifier()

    params = {
        "eta" : xbgeta,
        "gamma": XgammaBOOST,
        "max_depth" : deep_boost,
        "min_child_weight" : min_boost_weight,
        "subsample" : xgboosamplet,
        "colsample_bytree" : xgbytree,
        "ojective" : xgboojective,
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

