import numpy as np

def remove_features(x, features):
    """
    This method is used to remove features in one matrix of features.
    :param x: matrix of features
    :param features: indexes of features to remove
    :return: the new matrix of features without specific removed features
    """
    return np.delete(x, features, 1)

def standardize(x, testx):
    """
    This method is used to standardize training and testing X with the same mean and same standard deviation
    :param x: matrix X of training
    :param testx: matrix X of testing
    :return: the two matrix after standardization
    """
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    testx = testx - mean_x
    std_x = np.std(x, axis=0)
    x[:, std_x > 0] = x[:, std_x > 0] / std_x[std_x > 0]
    testx[:, std_x > 0] = testx[:, std_x > 0] / std_x[std_x > 0]
    return x, testx

def nan_to_mean(x, testx):
    """
    This method is used to replace -999 values with the mean of each column
    :param x: matrix X of training
    :param testx: matrix X of testing
    :return: the two matrix after substitution of each -999 value with the mean
    """
    x[np.where(x == -999)] = np.nan
    me = np.ma.array(x, mask=np.isnan(x)).mean(axis=0)
    means = np.ma.getdata(me)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(means, inds[1])
    testx[np.where(testx == -999)] = np.nan
    me = np.ma.array(testx, mask=np.isnan(testx)).mean(axis=0)
    means = np.ma.getdata(me)
    inds = np.where(np.isnan(testx))
    testx[inds] = np.take(means, inds[1])
    return x, testx

def adjust_features(x, testx):
    """
    This method is used to adjust, remove and add new features to the previous X matrix, both for training and testing
    :param x: matrix X of training
    :param testx: matrix X of testing
    :return: the two matrix with adjusted features
    """
    for jet in range(len(x)):
        inv_log_cols, x_delete_index = get_indexes(jet)
        x_train_inv_log_cols = np.log(1 / (1 + x[jet][:, inv_log_cols]))
        x[jet] = np.hstack((x[jet], x_train_inv_log_cols))
        x_test_inv_log_cols = np.log(1 / (1 + testx[jet][:, inv_log_cols]))
        testx[jet] = np.hstack((testx[jet], x_test_inv_log_cols))
        #x[jet], testx[jet] = add_cartesian_features(x, testx, jet)
        # Removing features
        x[jet] = remove_features(x[jet], x_delete_index)
        testx[jet] = remove_features(testx[jet], x_delete_index)
    return x, testx

def get_indexes(jet):
    """
    This method is used to get the indexes of the correct features to be removed and adjusted
    :param jet: number to identify the classifier (0-7)
    :return: the features to be adjusted, the features to be deleted.
    """
    if jet in (0, 1, 2, 3):
        if jet == 0:
            x_delete_index = [3, 4, 5, 11, 21, 22, 23, 24, 25, 26, 27, 28]
            inv_log_cols = [1, 6, 8, 9, 12, 15, 18, 20]
        if jet == 1:
            x_delete_index = [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29]
            inv_log_cols = [0, 2, 7, 9, 10, 13, 16, 19, 21]
        if jet == 2:
            x_delete_index = [3, 4, 5, 11, 21, 22, 24, 25, 26, 27]
            inv_log_cols = [1, 6, 8, 9, 12, 15, 18, 20]
        if jet == 3:
            x_delete_index = [4, 5, 6, 12, 22, 26, 27, 28]
            inv_log_cols = [0, 2, 7, 9, 10, 13, 16, 19, 21]
    elif jet in (4, 5, 6, 7):
        if jet in (4, 6):
            x_delete_index = [21]
            inv_log_cols = [1, 4, 6, 8, 9, 12, 15, 18, 20, 22, 25]
        elif jet in (5, 7):
            x_delete_index=[22]
            inv_log_cols = [0, 2, 5, 7, 9, 10, 13, 16, 19, 21, 23, 26]
    return inv_log_cols, x_delete_index

def add_cartesian_features(x, testx, jet):
    """
    This method is used to create the cartesian features of the momentum
    :param x: matrix X of training
    :param testx: matrix X of testing
    :param jet: number used to identify classifier
    :return: two new matrix X of training and testing with new cartesian features.
    """
    pri_tau_pt = 13
    pri_tau_eta = 14
    pri_tau_phi = 15
    pri_lep_pt = 16
    pri_lep_eta = 17
    pri_lep_phi = 18

    offset = 1 if jet % 2 == 0 else 0
    # TRAIN - TAU
    px_tau = x[jet][:, pri_tau_pt - offset] * np.sin(x[jet][:, pri_tau_phi - offset])
    py_tau = x[jet][:, pri_tau_pt - offset] * np.cos(x[jet][:, pri_tau_phi - offset])
    pz_tau = x[jet][:, pri_tau_pt - offset] * np.sinh(x[jet][:, pri_tau_eta - offset])
    # TRAIN - LEPT
    px_lep = x[jet][:, pri_lep_pt - offset] * np.sin(x[jet][:, pri_lep_phi - offset])
    py_lep = x[jet][:, pri_lep_pt - offset] * np.cos(x[jet][:, pri_lep_phi - offset])
    pz_lep = x[jet][:, pri_lep_pt - offset] * np.sinh(x[jet][:, pri_lep_eta - offset])
    # TEST - TAU
    px_taut = testx[jet][:, pri_tau_pt - offset] * np.sin(testx[jet][:, pri_tau_phi - offset])
    py_taut = testx[jet][:, pri_tau_pt - offset] * np.cos(testx[jet][:, pri_tau_phi - offset])
    pz_taut = testx[jet][:, pri_tau_pt - offset] * np.sinh(testx[jet][:, pri_tau_eta - offset])
    # TEST - LEPT
    px_lept = testx[jet][:, pri_lep_pt - offset] * np.sin(testx[jet][:, pri_lep_phi - offset])
    py_lept = testx[jet][:, pri_lep_pt - offset] * np.cos(testx[jet][:, pri_lep_phi - offset])
    pz_lept = testx[jet][:, pri_lep_pt - offset] * np.sinh(testx[jet][:, pri_lep_eta - offset])
    train_features = [px_tau, py_tau, pz_tau, px_lep, py_lep, pz_lep]
    test_features = [px_taut, py_taut, pz_taut, px_lept, py_lept, pz_lept]
    for feat in train_features:
        x[jet] = np.column_stack((x[jet], feat))
    for feat in test_features:
        testx[jet] = np.column_stack((testx[jet], feat))
    return x[jet], testx[jet]

def split_jets(train_x, train_y, test_x, test_y, idstest):
    """
    This method is used to split training and testing set into 8 different training and testing sets, based on:
        - JET number
        - presence of MASS
    :param train_x: features matrix X of training
    :param train_y: label vector y of training
    :param test_x: features matrix X of testing
    :param test_y: label vector y of testing
    :param idstest: indices of testing
    :return: an array of 8 different training and testing sets:
        0. jet 0 without mass
        1. jet 0 with mass
        2. jet 1 without mass
        3. jet 1 with mass
        4. jet 2 without mass
        5. jet 2 with mass
        6. jet 3 without mass
        7. jet 3 with mass
    """
    jet_number = 22
    jets = 4
    x_jets_train = []
    x_jets_test = []
    y_jets_train = []
    ids = []
    for jet in range(jets):
        x_jets_train.append(train_x[train_x[:, jet_number] == jet])
        x_jets_test.append(test_x[test_x[:, jet_number] == jet])
        ids_train = np.where([train_x[:, jet_number] == jet])[1]
        y_jets_train.append(train_y[ids_train])
        ids_test = np.where([test_x[:, jet_number] == jet])[1]
        ids.append(idstest[ids_test])
    mass_x_jets_train = []
    mass_y_jets_train = []
    mass_x_jets_test = []
    mass_ids_jets_test = []
    for jet in range(jets):
        indices_jet_nan_train = np.where([x_jets_train[jet][:, 0] == (-999)])[1]
        indices_jet_Nnan_train = np.where([x_jets_train[jet][:, 0] != (-999)])[1]
        indices_jet_nan_test = np.where([x_jets_test[jet][:, 0] == (-999)])[1]
        indices_jet_Nnan_test = np.where([x_jets_test[jet][:, 0] != (-999)])[1]
        mass_ids_jets_test.append(ids[jet][indices_jet_nan_test])
        mass_ids_jets_test.append(ids[jet][indices_jet_Nnan_test])
        x_jets_nan_train = x_jets_train[jet][indices_jet_nan_train]
        x_jets_nan_train = np.delete(x_jets_nan_train, 0, 1)
        mass_x_jets_train.append(x_jets_nan_train)
        mass_x_jets_train.append(x_jets_train[jet][indices_jet_Nnan_train])
        x_jets_nan_test = x_jets_test[jet][indices_jet_nan_test]
        x_jets_nan_test = np.delete(x_jets_nan_test, 0, 1)
        mass_x_jets_test.append(x_jets_nan_test)
        mass_x_jets_test.append(x_jets_test[jet][indices_jet_Nnan_test])
        mass_y_jets_train.append(y_jets_train[jet][indices_jet_nan_train])
        mass_y_jets_train.append(y_jets_train[jet][indices_jet_Nnan_train])
    return mass_x_jets_train, mass_x_jets_test, mass_y_jets_train, mass_ids_jets_test