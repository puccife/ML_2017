from utils.proj1_helpers import load_csv_data, predict_labels, create_csv_submission, cross_validation_visualization, sort_predictions
from utils.preprocessing import adjust_features, nan_to_mean, standardize, split_jets
from utils.implementations import ridge_regression, logistic_regression, build_k_indices, cross_validation
import time
import numpy as np
# PATHS for files
#   - train.csv: dataset used to train the model
#   - test.csv: dataset used to test the model
#   - prediction.csv dataset predicted by the model
train_path = './data/train.csv'
test_path = './data/test.csv'
OUTPUT = 'jet_pred/prediction.csv'

# Loading datasets
start_time = time.time()
print("Loading datasets")
train_y, train_x, train_ids = load_csv_data(train_path)
test_y, test_x, idstest_ids = load_csv_data(test_path)
print("Datasets loaded in: " + str(time.time() - start_time))

# Splitting dataset into 8 datasets to use 8 different classifiers
x_jets_train, \
x_jets_test, \
y_jets_train, \
ids = split_jets(train_x, train_y, test_x, test_y, idstest_ids)

# Adjusting features of the dataset
x_jets_train, x_jets_test = adjust_features(x_jets_train, x_jets_test)

# PARAMS for k-fold cross validation
seed = 33
k_fold = 8
jet_to_train = 4
degree = 3
lambdas = np.logspace(-9, -1, 30)

# Standardizing dataset
x_jets_train[jet_to_train], x_jets_test[jet_to_train] = standardize(x_jets_train[jet_to_train], x_jets_test[jet_to_train])

k_indices = build_k_indices(y_jets_train[jet_to_train], k_fold, seed)
rmse_tr = []
rmse_te = []
best_loss = 50000
best_lambda = 0
for lambda_ in lambdas:
    temp_tr = np.zeros(k_fold)
    temp_te = np.zeros(k_fold)
    for k in range(k_fold):
        tr_loss, te_loss, ws = cross_validation(y_jets_train[jet_to_train], x_jets_train[jet_to_train], k_indices, k, lambda_, degree)
        temp_tr[k] = tr_loss
        temp_te[k] = te_loss
    print(np.mean(temp_te))
    print(np.mean(temp_tr))
    if np.mean(temp_te) < best_loss:
        best_loss = np.mean(temp_te)
        best_lambda = lambda_
    print("After lambdas iteration, the best lambda is : " + str(best_lambda) + " for Lambda : " + str(lambda_) + " with best loss = " + str(best_loss))
    rmse_tr.append(np.mean(temp_tr))
    rmse_te.append(np.mean(temp_te))

# Saving cross validation visualization as png file named cross_validation.png
cross_validation_visualization(lambdas, rmse_tr, rmse_te)