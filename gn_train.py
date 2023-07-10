import pandas as pd
import numpy as np
import pickle as pkl

TRAIN_PATH = "train_gn.csv"
PARAMETERS_PATH = "parameters_gn.pkl"


def show_array(category_label, array, labels):
    print(f"\t{category_label} -> ", end="")
    for i in range(len(array)):
        print(f"{labels[i]}:{array[i]: >7.4f}     ", end="")
    print()


train_df = pd.read_csv(TRAIN_PATH)
X_df = train_df.iloc[:, :-1]
Y_df = train_df.iloc[:, -1]
n = len(X_df)
d = len(X_df.columns)
print(f"Read {n} samples with {d} attributes from {TRAIN_PATH}")

## Your code here

class_label = np.sort(Y_df.unique())
# Get the DF for each class
splitted_class = {}
class_total = {}

for w in range(len(class_label)):
    DF = "DF_" + class_label[w]
    DF = train_df.loc[train_df["Y"] == class_label[w]]
    df_size = len(DF)
    splitted_class[class_label[w]] = DF
    class_total[class_label[w]] = df_size

# Get the total number of each elements to calculate Priors
priors = {}
print("Priors:")
for k in range(len(class_label)):
    prior = (int(class_total[class_label[k]]) / n) * 100
    print(f"\t {class_label[k]}: {prior:.1f}%")
    priors[class_label[k]] = round(prior, 1)

# Getting the features label
features_labels = train_df.columns[0:4]
class_category_mean = (
    {}
)  # Using this to hole the general disctionary for class and mean of each column
class_category_stdvs = {}
for k in range(len(splitted_class)):
    category = class_label[k]
    means_category = {}
    stds_category = {}
    for j in range(len(features_labels)):
        column_name = features_labels[j]
        means = splitted_class[category][column_name].mean()
        means_category[features_labels[j]] = f"{means:7.4f}"
        stdvs = splitted_class[category][column_name].std()
        stds_category[features_labels[j]] = f"{stdvs:7.4f}"

    class_category_mean[category] = means_category
    class_category_stdvs[category] = stds_category

class_category_mean
category_label = ["Means", "Stdvs"]
for i in range(len(class_label)):
    print(f"\n{class_label[i]}:")
    print(f"\tMeans ->", end="")
    for k in range(len(features_labels)):
        print(
            f" {features_labels[k]}: {class_category_mean[class_label[i]][features_labels[k]]}   ",
            end="",
        )
    print(f"\n\tStdvs ->", end="")
    for k in range(len(features_labels)):
        print(
            f" {features_labels[k]}: {class_category_stdvs[class_label[i]][features_labels[k]]}   ",
            end="",
        )
# store the labels, the priors, the means, and the standard deviations in parameters gn.pkl.
parameters = {}  # Creating a dictionary to hold the scaler and the model(logreg)
parameters["labels"] = features_labels
parameters["priors"] = priors
parameters["means"] = class_category_mean
parameters["stdvs"] = class_category_stdvs
pkl.dump(parameters, open(PARAMETERS_PATH, "wb"))  # Save the parameters to pickle file
print()

print(f"Wrote parameters to {PARAMETERS_PATH}")
