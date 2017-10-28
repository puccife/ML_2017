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
    # Create inverse log values of features which are positive in value.
    for jet in range(len(x)):
        if jet in (0,1,2,3):
            if jet == 0:
                x_delete_index = [3, 4, 5, 11, 22, 23, 24, 25, 26, 27, 28]
                inv_log_cols = [1, 6, 8, 9, 12, 15, 18, 20]
            if jet == 1:
                x_delete_index = [4, 5, 6, 12, 23, 24, 25, 26, 27, 28, 29]
                inv_log_cols = [0, 2, 7, 9, 10, 13, 16, 19, 21]
            if jet == 2:
                x_delete_index = [3, 4, 5, 11, 25, 26, 27]
                inv_log_cols = [1, 6, 8, 9, 12, 15, 18, 20]
            if jet == 3:
                x_delete_index = [4, 5, 6, 12, 26, 27, 28]
                inv_log_cols = [0, 2, 7, 9, 10, 13, 16, 19, 21]
        elif jet in (4,5,6,7):
            if jet in (4,6):
                inv_log_cols = [1, 4, 6, 8, 9, 12, 15, 18, 20, 22, 25]
            elif jet in (5,7):
                inv_log_cols = [0, 2, 5, 7, 9, 10, 13, 16, 19, 21, 23, 26]
            x_delete_index = []

        x_train_inv_log_cols = np.log(1 / (1 + x[jet][:, inv_log_cols]))
        x[jet] = np.hstack((x[jet], x_train_inv_log_cols))
        x_test_inv_log_cols = np.log(1 / (1 + testx[jet][:, inv_log_cols]))
        testx[jet] = np.hstack((testx[jet], x_test_inv_log_cols))

        x[jet] = remove_features(x[jet], x_delete_index)
        testx[jet] = remove_features(testx[jet], x_delete_index)


        # Preprocessing dataset
        # x[jet], testx[jet] = nan_to_mean(x[jet], testx[jet])
        # offset = 0 if jet % 2 == 0 else 1
        # # TRAIN - TAU
        # px_tau = x[jet][:, pri_tau_pt-offset] * np.sin(x[jet][:, pri_tau_phi-offset])
        # py_tau = x[jet][:, pri_tau_pt-offset] * np.cos(x[jet][:, pri_tau_phi-offset])
        # pz_tau = x[jet][:, pri_tau_pt-offset] * np.sinh(x[jet][:, pri_tau_eta-offset])
        # # TRAIN - LEPT
        # px_lep = x[jet][:, pri_lep_pt-offset] * np.sin(x[jet][:, pri_lep_phi-offset])
        # py_lep = x[jet][:, pri_lep_pt-offset] * np.cos(x[jet][:, pri_lep_phi-offset])
        # pz_lep = x[jet][:, pri_lep_pt-offset] * np.sinh(x[jet][:, pri_lep_eta-offset])
        # # TEST - TAU
        # px_taut = testx[jet][:, pri_tau_pt-offset] * np.sin(testx[jet][:, pri_tau_phi-offset])
        # py_taut = testx[jet][:, pri_tau_pt-offset] * np.cos(testx[jet][:, pri_tau_phi-offset])
        # pz_taut = testx[jet][:, pri_tau_pt-offset] * np.sinh(testx[jet][:, pri_tau_eta-offset])
        # # TEST - LEPT
        # px_lept = testx[jet][:, pri_lep_pt-offset] * np.sin(testx[jet][:, pri_lep_phi-offset])
        # py_lept = testx[jet][:, pri_lep_pt-offset] * np.cos(testx[jet][:, pri_lep_phi-offset])
        # pz_lept = testx[jet][:, pri_lep_pt-offset] * np.sinh(testx[jet][:, pri_lep_eta-offset])
        #
        # train_features = [px_tau,py_tau,pz_tau,
        #                   px_lep,py_lep,pz_lep]
        # test_features = [px_taut, py_taut,pz_taut,
        #                   px_lept,py_lept,pz_lept]
        # for feat in train_features:
        #     x[jet] = np.column_stack((x[jet], feat))
        # for feat in test_features:
        #     testx[jet] = np.column_stack((testx[jet], feat))

    return x, testx

def split_jets(train_x, train_y, test_x, test_y, idstest):
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

    # FOR EACH JET
    # FIRST WITHOUT MASS
    # SECOND WITH MASS

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
    # return x_jets_train, x_jets_test, y_jets_train, ids