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


def project_model(training_data, random_seed):

    stratify = 0 #switch whether or not to stratify target_variable
    target_variable = "bank_account" #maybe we use this for the next project too

    #read data from csv
    #data is already hot encoded

