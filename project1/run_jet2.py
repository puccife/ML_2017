from utils.proj1_helpers import load_csv_data, predict_labels, create_csv_submission, cross_validation_visualization
from utils.preprocessing import adjust_features, nan_to_mean, standardize, split_jets
from utils.train_utils import build_poly
from utils.implementations import ridge_regression, logistic_regression, build_k_indices, cross_validation
import shutil
import glob
import numpy as np
import time
import sys, csv ,operator
train_path = './data/train.csv'
test_path = './data/test.csv'
start_time = time.time()
print("Loading datasets")
train_y, train_x, train_ids = load_csv_data(train_path)
print("Train dataset loaded in " + str(time.time() - start_time))
start_time = time.time()
test_y, test_x, idstest_ids = load_csv_data(test_path)
print("Test dataset loaded in " + str(time.time() - start_time))
train_x, test_x = nan_to_mean(train_x, test_x)
train_x, test_x = adjust_features(train_x, test_x)
x_jets_0_train, x_jets_1_train, x_jets_2_train, x_jets_3_train, \
x_jets_0_test, x_jets_1_test, x_jets_2_test, x_jets_3_test, \
y_jets_0_train, y_jets_1_train, y_jets_2_train, y_jets_3_train, \
y_jets_0_test, y_jets_1_test, y_jets_2_test, y_jets_3_test, \
idstest_0, idstest_1, idstest_2, idstest_3 = split_jets(train_x, train_y, test_x, test_y, idstest_ids)
x_jets_0_train, x_jets_0_test = standardize(x_jets_0_train,x_jets_0_test)
x_jets_1_train, x_jets_1_test = standardize(x_jets_1_train,x_jets_1_test)
x_jets_2_train, x_jets_2_test = standardize(x_jets_2_train,x_jets_2_test)
x_jets_3_train, x_jets_3_test = standardize(x_jets_3_train,x_jets_3_test)
# ------------------------- LOCAL TEST -------------------------------
seed = 1
#
# degree_for_0 = 1
# degree_for_1 = 3
# degree_for_2 = 3
# degree_for_3 = 3
# degree = 3
# k_fold = 8
# lambdas = np.logspace(-9, -1, 30)
# k_indices = build_k_indices(y_jets_3_train, k_fold, seed)
# rmse_tr = []
# rmse_te = []
# best_loss = 999
# best_lambda = 0
# for lambda_ in lambdas:
#     temp_tr = np.zeros(k_fold)
#     temp_te = np.zeros(k_fold)
#     for k in range(k_fold):
#         tr_loss, te_loss, ws = cross_validation(y_jets_3_train, x_jets_3_train, k_indices, k, lambda_, degree)
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
degree = 1
px_train_0 = build_poly(degree=degree,x=x_jets_0_train)
px_test_0 = build_poly(degree=degree,x=x_jets_0_test)
degree = 3
px_train_1 = build_poly(degree=degree,x=x_jets_1_train)
px_test_1 = build_poly(degree=degree,x=x_jets_1_test)
px_train_2 = build_poly(degree=degree,x=x_jets_2_train)
px_test_2 = build_poly(degree=degree,x=x_jets_2_test)
px_train_3 = build_poly(degree=degree,x=x_jets_3_train)
px_test_3 = build_poly(degree=degree,x=x_jets_3_test)
ws_0, loss0 = ridge_regression(lambda_=6.7233575365e-09, tx=px_train_0, y=y_jets_0_train)
ws_1, loss1 = ridge_regression(lambda_=4.89390091848e-05, tx=px_train_1, y=y_jets_1_train)
ws_2, loss2 = ridge_regression(lambda_=4.52035365636e-08, tx=px_train_2, y=y_jets_2_train)
ws_3, loss3 = ridge_regression(lambda_=6.7233575365e-09, tx=px_train_3, y=y_jets_3_train)
# w_sgd0, loss_sgd0 = logistic_regression(y_jets_0_train, px_train_0, None, 200, gamma)
# w_sgd1, loss_sgd1 = logistic_regression(y_jets_1_train, px_train_1, None, 200, gamma)
# w_sgd2, loss_sgd2 = logistic_regression(y_jets_2_train, px_train_2, None, 200, gamma)
# w_sgd3, loss_sgd3 = logistic_regression(y_jets_3_train, px_train_3, None, 200, gamma)
y_pred_0 = predict_labels(ws_0, px_test_0)
y_pred_1 = predict_labels(ws_1, px_test_1)
y_pred_2 = predict_labels(ws_2, px_test_2)
y_pred_3 = predict_labels(ws_3, px_test_3)
# ---------------------- CREATE SUBMISSION --------------------------------
create_csv_submission(idstest_0, y_pred_0, 'jet_pred/prediction_0.csv')
create_csv_submission(idstest_1, y_pred_1, 'jet_pred/prediction_1.csv')
create_csv_submission(idstest_2, y_pred_2, 'jet_pred/prediction_2.csv')
create_csv_submission(idstest_3, y_pred_3, 'jet_pred/prediction_3.csv')

interesting_files = glob.glob("jet_pred/*.csv")

with open('someoutputfile_1.csv', 'wb') as outfile:
    for i, fname in enumerate(interesting_files):
        with open(fname, 'rb') as infile:
            if i != 0:
                infile.readline()  # Throw away header on all but first file
            # Block copy rest of file from input to output without parsing
            shutil.copyfileobj(infile, outfile)
            print(fname + " has been imported.")

data = csv.reader(open('someoutputfile_1.csv'),delimiter=',')
sortedlist = sorted(data, key=operator.itemgetter(0))
del sortedlist[len(sortedlist)-1]
with open("Sorted_output.csv", "w") as f:
    fileWriter = csv.writer(f, delimiter=',')
    fileWriter.writerow(["Id", "Prediction"])
    for row in sortedlist:
        fileWriter.writerow([int(row[0]), int(row[1])])