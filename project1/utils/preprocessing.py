import numpy as np

# PHI features
pri_jet_phi = 25
pri_sub_phi = 28
# DERIVATIVE
DER_mass_MMC = 0
DER_mass_transverse_met_lep = 1
DER_deltaeta_jet_jet = 4
DER_mass_jet_jet = 5
DER_prodeta_jet_jet = 6
DER_lep_eta_centrality = 12
met_phi = 20
# PRIMITIVE
pri_tau_pt = 13
pri_tau_eta = 14
pri_tau_phi = 15
pri_lep_pt = 16
pri_lep_eta = 17
pri_lep_phi = 18
def remove_features(x, features):
    return np.delete(x, features, 1)

def standardize(x, testx):
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    testx = testx - mean_x
    std_x = np.std(x, axis=0)
    x[:, std_x > 0] = x[:, std_x > 0] / std_x[std_x > 0]
    testx[:, std_x > 0] = testx[:, std_x > 0] / std_x[std_x > 0]
    return x, testx

def nan_to_mean_jet(x_jets_train_matrix, x_jets_test_matrix):
    for jet_matrix in x_jets_train_matrix:
        jet_matrix[np.where(jet_matrix == -999)] = np.nan
        me = np.ma.array(jet_matrix, mask=np.isnan(jet_matrix)).mean(axis=0)
        means = np.ma.getdata(jet_matrix)
        inds = np.where(np.isnan(jet_matrix))
        jet_matrix[inds] = np.take(means, inds[1])
    for jet_matrix in x_jets_test_matrix:
        jet_matrix[np.where(jet_matrix == -999)] = np.nan
        me = np.ma.array(jet_matrix, mask=np.isnan(jet_matrix)).mean(axis=0)
        means = np.ma.getdata(jet_matrix)
        inds = np.where(np.isnan(jet_matrix))
        jet_matrix[inds] = np.take(means, inds[1])
    return x_jets_train_matrix, x_jets_test_matrix

def nan_to_mean(x, testx):
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
    square_mass = np.sqrt(x[:, DER_mass_MMC])
    square_mass_t = np.sqrt(x[:, DER_mass_transverse_met_lep])
    px_tau = x[:, pri_tau_pt] * np.sin(x[:, pri_tau_phi])
    py_tau = x[:, pri_tau_pt] * np.cos(x[:, pri_tau_phi])
    pz_tau = x[:, pri_tau_pt] * np.sinh(x[:, pri_tau_eta])
    mod_tau = x[:, pri_tau_pt] * np.cosh(x[:, pri_tau_eta])
    # magnitude_tau = np.sqrt(np.square(px_tau) + np.square(py_tau) + np.square(pz_tau))
    # magnitude_tau_square = (np.square(px_tau) + np.square(py_tau) + np.square(pz_tau))
    # TRAIN - LEPT
    px_lep = x[:, pri_lep_pt] * np.sin(x[:, pri_lep_phi])
    py_lep = x[:, pri_lep_pt] * np.cos(x[:, pri_lep_phi])
    pz_lep = x[:, pri_lep_pt] * np.sinh(x[:, pri_lep_eta])
    mod_lep = x[:, pri_lep_pt] * np.cosh(x[:, pri_lep_eta])
    # magnitude_lep = np.sqrt(np.square(px_lep) + np.square(py_lep) + np.square(pz_lep))
    # magnitude_lep_square = (np.square(px_lep) + np.square(py_lep) + np.square(pz_lep))
    # TEST - TAU
    tsquare_mass = np.sqrt(testx[:, DER_mass_MMC])
    tsquare_mass_t = np.sqrt(testx[:, DER_mass_transverse_met_lep])
    px_taut = testx[:, pri_tau_pt] * np.sin(testx[:, pri_tau_phi])
    py_taut = testx[:, pri_tau_pt] * np.cos(testx[:, pri_tau_phi])
    pz_taut = testx[:, pri_tau_pt] * np.sinh(testx[:, pri_tau_eta])
    mod_taut = testx[:, pri_tau_pt] * np.cosh(testx[:, pri_tau_eta])
    # magnitude_taut = np.sqrt(np.square(px_taut) + np.square(py_taut) + np.square(pz_taut))
    # magnitude_taut_square = (np.square(px_taut) + np.square(py_taut) + np.square(pz_taut))
    # TEST - LEPT
    px_lept = testx[:, pri_lep_pt] * np.sin(testx[:, pri_lep_phi])
    py_lept = testx[:, pri_lep_pt] * np.cos(testx[:, pri_lep_phi])
    pz_lept = testx[:, pri_lep_pt] * np.sinh(testx[:, pri_lep_eta])
    mod_lept = testx[:, pri_lep_pt] * np.cosh(testx[:, pri_lep_eta])
    # magnitude_lept = np.sqrt(np.square(px_lept) + np.square(py_lept) + np.square(pz_lept))
    # magnitude_lept_square = (np.square(px_lept) + np.square(py_lept) + np.square(pz_lept))
    train_features = [px_tau,py_tau,pz_tau,mod_tau,
                      px_lep,py_lep,pz_lep,mod_lep, square_mass, square_mass_t]
    test_features = [px_taut, py_taut,pz_taut,mod_taut,
                      px_lept,py_lept,pz_lept,mod_lept, tsquare_mass, tsquare_mass_t]
    for feat in train_features:
        x = np.column_stack((x, feat))
    for feat in test_features:
        testx = np.column_stack((testx, feat))
    return x, testx

def split_jets(train_x, train_y, test_x, test_y, idstest):
    jet_number = 22

    x_jets_0_train = train_x[train_x[:, jet_number] == 0]
    x_jets_1_train = train_x[train_x[:, jet_number] == 1]
    x_jets_2_train = train_x[train_x[:, jet_number] == 2]
    x_jets_3_train = train_x[train_x[:, jet_number] == 3]

    x_jets_0_test = test_x[test_x[:, jet_number] == 0]
    x_jets_1_test = test_x[test_x[:, jet_number] == 1]
    x_jets_2_test = test_x[test_x[:, jet_number] == 2]
    x_jets_3_test = test_x[test_x[:, jet_number] == 3]

    indices_x_0_train = np.where([train_x[:, jet_number] == 0])[1]
    indices_x_1_train = np.where([train_x[:, jet_number] == 1])[1]
    indices_x_2_train = np.where([train_x[:, jet_number] == 2])[1]
    indices_x_3_train = np.where([train_x[:, jet_number] == 3])[1]

    y_jets_0_train = train_y[indices_x_0_train]
    y_jets_1_train = train_y[indices_x_1_train]
    y_jets_2_train = train_y[indices_x_2_train]
    y_jets_3_train = train_y[indices_x_3_train]

    indices_x_0_test = np.where([test_x[:, jet_number] == 0])[1]
    indices_x_1_test = np.where([test_x[:, jet_number] == 1])[1]
    indices_x_2_test = np.where([test_x[:, jet_number] == 2])[1]
    indices_x_3_test = np.where([test_x[:, jet_number] == 3])[1]

    y_jets_0_test = test_y[indices_x_0_test]
    y_jets_1_test = test_y[indices_x_1_test]
    y_jets_2_test = test_y[indices_x_2_test]
    y_jets_3_test = test_y[indices_x_3_test]

    idstest_0 = idstest[indices_x_0_test]
    idstest_1 = idstest[indices_x_1_test]
    idstest_2 = idstest[indices_x_2_test]
    idstest_3 = idstest[indices_x_3_test]

    x_delete_index_0 = [0,1,4,5,6,12,15,18,20,23,24,25,26,27,28,29]
    x_delete_index_1 = [0,1,4,5,6,12,15,18,20,25,26,27,28]
    x_delete_index_2 = [0,1,15,18,20,25,28]
    x_delete_index_3 = [0,1,15,18,20,25,28]

    x_jets_0_train = remove_features(x_jets_0_train, x_delete_index_0)
    x_jets_0_test = remove_features(x_jets_0_test, x_delete_index_0)
    x_jets_1_train = remove_features(x_jets_1_train, x_delete_index_1)
    x_jets_1_test = remove_features(x_jets_1_test, x_delete_index_1)
    x_jets_2_train = remove_features(x_jets_2_train, x_delete_index_2)
    x_jets_2_test = remove_features(x_jets_2_test, x_delete_index_2)
    x_jets_3_train = remove_features(x_jets_3_train, x_delete_index_3)
    x_jets_3_test = remove_features(x_jets_3_test, x_delete_index_3)

    return x_jets_0_train, x_jets_1_train, x_jets_2_train, x_jets_3_train,\
           x_jets_0_test, x_jets_1_test, x_jets_2_test, x_jets_3_test, \
           y_jets_0_train, y_jets_1_train, y_jets_2_train, y_jets_3_train, \
           y_jets_0_test, y_jets_1_test, y_jets_2_test, y_jets_3_test,\
           idstest_0, idstest_1, idstest_2, idstest_3