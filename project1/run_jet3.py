from utils.proj1_helpers import load_csv_data, predict_labels, create_csv_submission, cross_validation_visualization, sort_predictions
from utils.preprocessing import adjust_features, nan_to_mean, standardize, split_jets
from utils.train_utils import build_poly, get_lambda, get_degree
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

train_x, test_x = adjust_features(x_jets_train, x_jets_test)

px_train = []
px_test = []
ws = []
losses = []
preds = []

# Training model.
#   - Every function uses different parameters depending on the classifier
for i in range(len(x_jets_train)):

    # Getting correct Lambda
    lambda_ = get_lambda(i)
    # Getting correct Degree
    degree = get_degree(i)

    # Standardizing dataset
    x_jets_train[i], x_jets_test[i] = standardize(x_jets_train[i], x_jets_test[i])

    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame(x_jets_train[i])
    df.hist()
    plt.show()

    # Building polynomial features
    px_train.append(build_poly(degree=degree,x=x_jets_train[i]))
    px_test.append(build_poly(degree=degree,x=x_jets_test[i]))

    # Training model
    w, loss = ridge_regression(lambda_=lambda_, tx=px_train[i], y=y_jets_train[i])
    print(loss)
    ws.append(w)
    losses.append(loss)

    # Predicting labels
    preds.append(predict_labels(w, px_test[i]))


# ------------------------- LOCAL TEST -------------------------------
# seed = 33
# k_fold = 8
# jet_to_train = 5
# degree = 5
#
# x_jets_train[jet_to_train], x_jets_test[jet_to_train] = standardize(x_jets_train[jet_to_train], x_jets_test[jet_to_train])
#
# lambdas = np.logspace(-6, -3, 100)
# k_indices = build_k_indices(y_jets_train[jet_to_train], k_fold, seed)
# rmse_tr = []
# rmse_te = []
# best_loss = 999
# best_lambda = 0
# for lambda_ in lambdas:
#     temp_tr = np.zeros(k_fold)
#     temp_te = np.zeros(k_fold)
#     for k in range(k_fold):
#         tr_loss, te_loss, ws = cross_validation(y_jets_train[jet_to_train], x_jets_train[jet_to_train], k_indices, k, lambda_, degree)
#         temp_tr[k] = tr_loss
#         temp_te[k] = te_loss
#     print(np.mean(temp_te))
#     print(np.mean(temp_tr))
#     if np.mean(temp_te) < best_loss:
#         best_loss = np.mean(temp_te)
#         best_lambda = lambda_
#     print("After lambdas iteration, the best lambda is : " + str(best_lambda) + " for Lambda : " + str(lambda_) + " with best loss = " + str(best_loss))
#     rmse_tr.append(np.mean(temp_tr))
#     rmse_te.append(np.mean(temp_te))
# cross_validation_visualization(lambdas, rmse_tr, rmse_te)
# ------------------------- ONLINE TEST ----------------------------------
# w_sgd0, loss_sgd0 = logistic_regression(y_jets_0_train, px_train_0, None, 200, gamma)
# w_sgd1, loss_sgd1 = logistic_regression(y_jets_1_train, px_train_1, None, 200, gamma)
# w_sgd2, loss_sgd2 = logistic_regression(y_jets_2_train, px_train_2, None, 200, gamma)
# w_sgd3, loss_sgd3 = logistic_regression(y_jets_3_train, px_train_3, None, 200, gamma)
# ----------------q------ CREATE SUBMISSION --------------------------------

# Sorting predictions
ids, preds = sort_predictions(ids, preds)
print("Creating submission")
# Creating output files
create_csv_submission(ids, preds, OUTPUT)
print("Created submission")