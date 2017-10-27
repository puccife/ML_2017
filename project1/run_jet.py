from utils.proj1_helpers import load_csv_data, predict_labels, create_csv_submission, cross_validation_visualization
from utils.preprocessing import adjust_features, nan_to_mean, standardize, split_jets
from utils.train_utils import split_data, build_poly
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
# train_x, train_y, test_x, test_y = split_data(train_x, train_y, 0.8, seed=1)
degree = 3
px_train_0 = build_poly(degree=degree,x=x_jets_0_train)
px_test_0 = build_poly(degree=degree,x=x_jets_0_test)
px_train_1 = build_poly(degree=degree,x=x_jets_1_train)
px_test_1 = build_poly(degree=degree,x=x_jets_1_test)
px_train_2 = build_poly(degree=degree,x=x_jets_2_train)
px_test_2 = build_poly(degree=degree,x=x_jets_2_test)
px_train_3 = build_poly(degree=degree,x=x_jets_3_train)
px_test_3 = build_poly(degree=degree,x=x_jets_3_test)
# ws, loss = ridge_regression(lambda_=0.00067233575365, tx=train_px, y=train_y)
# best_gamma = 0.000000001 # 1e-9 ====> 76.5 76.37//
# best_gamma = 0.00000001 # 1e-8 ====> 80.2 // 80.2 // 80.2135 // 79.98 // 80.4595 // 80.60 // 80.789
# good_gamma = 0.000000005 # 5e-9 ====> 79.5 // 79.6
gamma = 0.00000001
# ------------------------- LOCAL TEST -------------------------------
# seed = 1
# degree = 3
# k_fold = 10
# lambdas = np.logspace(-8, 0, 30)
# k_indices = build_k_indices(y_jets_0_train, k_fold, seed)
# rmse_tr = []
# rmse_te = []
# best_loss = 999
# for k in range(k_fold):
#     temp_tr = []
#     temp_te = []
#     best_test_ws = []
#     for lambda_ in lambdas:
#         tr_loss, te_loss, ws = cross_validation(y_jets_0_train, x_jets_0_train, k_indices, k, lambda_, degree)
#         if(te_loss < best_loss):
#             best_loss = te_loss
#             best_lambda = lambda_
#         temp_tr.append(tr_loss)
#         temp_te.append(te_loss)
#         # print("Lambda = " + str(lambda_) + " tr_loss = " + str(tr_loss) + " te_loss = " + str(te_loss))
#     print("After lambdas iteration, the best lambda is : " + str(best_lambda) + " for k-fold : " + str(k) + " with best loss = " + str(best_loss))
#     best_test_ws.append(lambda_)
#     rmse_tr.append(temp_tr)
#     rmse_te.append(temp_te)
# rmse_tr = np.matrix(rmse_tr)
# rmse_tr = np.mean(rmse_tr, axis=0)
# rmse_tr = np.reshape(rmse_tr, (len(lambdas),-1))
# rmse_te = np.matrix(rmse_te)
# rmse_te = np.mean(rmse_te, axis=0)
# rmse_te = np.reshape(rmse_te, (len(lambdas),-1))
# cross_validation_visualization(lambdas, rmse_tr, rmse_te)
#
# best_lambda_ever = 0.00067233575365

# train_x_0, train_y_0, test_x_0, test_y_0 = split_data(px_train_0, y_jets_0_train, 0.8, seed=1)
# train_x_1, train_y_1, test_x_1, test_y_1 = split_data(px_train_1, y_jets_1_train, 0.8, seed=1)
# train_x_2, train_y_2, test_x_2, test_y_2 = split_data(px_train_2, y_jets_2_train, 0.8, seed=1)
# train_x_3, train_y_3, test_x_3, test_y_3 = split_data(px_train_3, y_jets_3_train, 0.8, seed=1)
# ws_0, loss0 = ridge_regression(lambda_=0.0001, tx=train_x_0, y=train_y_0)
# ws_1, loss1 = ridge_regression(lambda_=0.00067233575365, tx=train_x_1, y=train_y_1)
# ws_2, loss2 = ridge_regression(lambda_=0.0001, tx=train_x_2, y=train_y_2)
# ws_3, loss3 = ridge_regression(lambda_=0.0001, tx=train_x_3, y=train_y_3)
# ws_0, loss_sgd0 = logistic_regression(train_y_0, train_x_0, None, 20000, gamma)
# ws_1, loss_sgd1 = logistic_regression(train_y_1, train_x_1, None, 20000, gamma)
# ws_2, loss_sgd2 = logistic_regression(train_y_2, train_x_2, None, 20000, gamma)
# ws_3, loss_sgd3 = logistic_regression(train_y_3, train_x_3, None, 20000, gamma)
# y_pred_0 = predict_labels(ws_0, test_x_0)
# y_pred_1 = predict_labels(ws_1, test_x_1)
# y_pred_2 = predict_labels(ws_2, test_x_2)
# y_pred_3 = predict_labels(ws_3, test_x_3)
# accuracy_0 = 1 - np.mean( y_pred_0 != test_y_0 )
# print("Accuracy: " + str(accuracy_0) + "%")
# accuracy_1 = 1 - np.mean( y_pred_1 != test_y_1 )
# print("Accuracy: " + str(accuracy_1) + "%")
# accuracy_2 = 1 - np.mean( y_pred_2 != test_y_2 )
# print("Accuracy: " + str(accuracy_2) + "%")
# accuracy_3 = 1 - np.mean( y_pred_3 != test_y_3 )
# print("Accuracy: " + str(accuracy_3) + "%")












# ------------------------- ONLINE TEST ----------------------------------

ws_0, loss0 = ridge_regression(lambda_=1.08263673387e-05, tx=px_train_0, y=y_jets_0_train)
ws_1, loss1 = ridge_regression(lambda_=6.7233575365e-08, tx=px_train_1, y=y_jets_1_train)
ws_2, loss2 = ridge_regression(lambda_=1e-08, tx=px_train_2, y=y_jets_2_train)
ws_3, loss3 = ridge_regression(lambda_=3.03919538231e-06, tx=px_train_3, y=y_jets_3_train)

# w_sgd0, loss_sgd0 = logistic_regression(y_jets_0_train, px_train_0, None, 200, gamma)
# w_sgd1, loss_sgd1 = logistic_regression(y_jets_1_train, px_train_1, None, 200, gamma)
# w_sgd2, loss_sgd2 = logistic_regression(y_jets_2_train, px_train_2, None, 200, gamma)
# w_sgd3, loss_sgd3 = logistic_regression(y_jets_3_train, px_train_3, None, 200, gamma)
#
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
#allFiles = glob.glob(path + "/*.csv")
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