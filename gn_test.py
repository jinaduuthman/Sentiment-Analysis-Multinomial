import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

TEST_PATH = "test_gn.csv"
CONFUSION_PATH = "confusion_gn.png"
PLOT_PATH = "confidence_gn.png"
PARAMETERS_PATH = "parameters_gn.pkl"

## Your code here

print(f"Read parameters from {PARAMETERS_PATH}")
file = open(PARAMETERS_PATH, "rb")
parameters = pkl.load(file)
labels = parameters["labels"]
priors = parameters["priors"]
means = parameters["means"]
stdvs = parameters["stdvs"]

### Your code here

test_df = pd.read_csv(TEST_PATH)
X = test_df.iloc[:, :-1].to_numpy(dtype=np.float64)
n = X.shape[0]
Y_df = test_df.iloc[:, -1]
print(f"Read {len(test_df)} rows from {TEST_PATH}")

## Your code here


print(f'Saved to "{PLOT_PATH}".')
