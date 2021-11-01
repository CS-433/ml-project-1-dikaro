import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
from helpers_data import *

print("\n--------- Data Loading ---------")
DATA_TRAIN_PATH = '../Data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
features = np.genfromtxt("../Data/train.csv", usecols=np.arange(2,32), delimiter=",", dtype=str, max_rows=1)

print("\n--------- Data Preprocessing ---------")
pol_degree = 13
rem_inds = []
y_clean, tX_clean, ms, stds = preprocess_dataset(y, tX, features, pol_degree, rem_inds)

print("\nFinal Shape of Input Data = {sh}.format".format(sh=tX_clean.shape))

# Regularized Logistic Regression using MiniBatch-SGD
print("\n--------- Applying ML Algorithm ---------")
lambda_ = 1e-7
gamma = 1e-4
batch_size = 8000
max_iters = 100000
k_fold = 10
div = 100   # Used to get statistics for max_iters/div number of iterations (to reduce cost)
w = cross_validation_log_regr_reg_SGD(y_clean, tX_clean, gamma, lambda_, batch_size, max_iters, k_fold, div, pol_degree)


print("\n--------- Creating csv Submission ---------")

DATA_TEST_PATH = '../Data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

tX_test_clean = preprocess_test_dataset(tX_test, pol_degree, ms, stds, rem_inds)
y_pred = predict_labels_LogReg(w, tX_test_clean)
y_pred = decode_y(y_pred)

OUTPUT_PATH = '../Data/solution.csv'
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

