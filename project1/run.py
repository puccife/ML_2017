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

# Adjusting features of the dataset
x_jets_train, x_jets_test = adjust_features(x_jets_train, x_jets_test)

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

# Sorting predictions
ids, preds = sort_predictions(ids, preds)
print("Creating submission")
# Creating output files
create_csv_submission(ids, preds, OUTPUT)
print("Created submission")