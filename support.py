import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from lifelines.utils import concordance_index as ci
from lifelines.statistics import logrank_test

def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + 'Folder create successfully !')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' Folader is exist')
        return False

def cal_pval(time, pred):
    event = np.zeros_like(time)
    event[time > 0] = 1
    pred_median = np.median(pred)
    risk_group = np.zeros_like(pred)
    risk_group[pred > pred_median] = 1

    group_lowrisk_time = time[risk_group==0].copy()
    group_highrisk_time = time[risk_group==1].copy()
    group_lowrisk_event = event[risk_group==0].copy()
    group_highrisk_event = event[risk_group==1].copy()

    results = logrank_test(group_lowrisk_time, group_highrisk_time, event_observed_A=group_lowrisk_event , event_observed_B=group_highrisk_event)
    # results.print_summary()
    return results.p_value



def sort_data(X,Y):
    T = - np.abs(np.squeeze(np.array(Y)))
    sorted_idx = np.argsort(T)
    return sorted_idx, X[sorted_idx], Y[sorted_idx]
def get_omic_data(fea_filename, seed, nfold = 5, fold_num =0):
    X, Y, H = load_omic_data(fea_filename)
    train_X, train_Y, test_X, test_Y = split_data(spilt_seed=seed, fea=X, label=Y, nfold=nfold, fold_num=fold_num)
    return train_X, train_Y, test_X, test_Y



def label_extra(data, t_col="Time", e_col="Event"):
    X = data[[c for c in data.columns if c not in [t_col, e_col]]]
    Y = data[[t for t in data.columns if t in[t_col]]]
    Y.loc[data[e_col]==0] = -Y.loc[data[e_col]==0]
    return X.values,Y.values



def read_data(filename):
    train_data = pd.read_csv(filename + "train.csv")
    test_data = pd.read_csv(filename + "test.csv")
    train_X,train_Y = label_extra(train_data)
    test_X, test_Y = label_extra(test_data)
    return train_X, train_Y, test_X, test_Y



def load_omic_data(fea_filename):
    data_fea = pd.read_csv(fea_filename)
    headers = data_fea.columns.values.tolist()
    headers = headers[1:]
    headers = np.array(headers)
    time = data_fea.iloc[0,:].tolist()[1:]
    time = np.array(time)
    status = data_fea.iloc[1, :].tolist()[1:]
    status = np.array(status)
    data_fea = data_fea[2:]  ##delete label
    for i in range(len(time)):
        if status[i] == 0:
            time[i] = -time[i]
    data_fea = data_fea.drop('GeneSymbol', axis=1)
    data_fea = data_fea.values
    data_fea = np.transpose(data_fea)
    data_time = time.reshape(-1, 1)
    print(len(data_fea))
    print(len(data_time))
    return data_fea, data_time, headers


def plot_curve(curve_data, title="train epoch-Cindex curve", x_label="epoch", y_label="Cindex"):
    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(curve_data[0], curve_data[1], color='black', markerfacecolor='black', marker='o', markersize=1)
    plt.show()
    ###print sorted survival time:

def split_data(spilt_seed, fea, label, nfold = 5, fold_num = 0):
    kf = KFold(n_splits=nfold, shuffle=True, random_state=spilt_seed)
    ###对齐输入

    label_flat = label.flatten()
    censor_index = np.where(label_flat < 0)
    no_cen_index = np.where(label_flat >= 0)
    censor = label[[tuple(censor_index)]]
    censor_fea = fea[[tuple(censor_index)]]
    censor = censor[0]
    censor_fea = censor_fea[0]

    no_cen = label[[tuple(no_cen_index)]]
    nocen_fea = fea[[tuple(no_cen_index)]]
    no_cen = no_cen[0]
    nocen_fea = nocen_fea[0]
    num = 0
    for train_index, test_index in kf.split(censor_fea):
        train_X1 = censor_fea[train_index]
        train_Y1 = censor[train_index]
        test_X1 = censor_fea[test_index]
        test_Y1 = censor[test_index]
        if num == fold_num:
            break
        num += 1

    num = 0
    for train_index, test_index in kf.split(nocen_fea):
        train_X2 = nocen_fea[train_index]
        train_Y2 = no_cen[train_index]
        test_X2 = nocen_fea[test_index]
        test_Y2 = no_cen[test_index]
        if num == fold_num:
            break
        num += 1

    train_X = np.vstack((train_X1, train_X2))
    train_Y = np.vstack((train_Y1, train_Y2))
    test_X = np.vstack((test_X1, test_X2))
    test_Y = np.vstack((test_Y1, test_Y2))
    return train_X, train_Y, test_X, test_Y


def split_data_with_headers(spilt_seed, fea, label,headers, nfold = 5, fold_num = 0):
    kf = KFold(n_splits=nfold, shuffle=True, random_state=spilt_seed)
    ###对齐输入

    label_flat = label.flatten()
    censor_index = np.where(label_flat < 0)
    no_cen_index = np.where(label_flat >= 0)
    censor = label[[tuple(censor_index)]]
    censor_fea = fea[[tuple(censor_index)]]
    censor_headers = headers[[tuple(censor_index)]]
    censor = censor[0]
    censor_fea = censor_fea[0]
    censor_headers = censor_headers[0]
    censor_headers = censor_headers.reshape(((len(censor_headers), 1)))  # (3,1)

    no_cen = label[[tuple(no_cen_index)]]
    nocen_fea = fea[[tuple(no_cen_index)]]
    no_cen_headers = headers[[tuple(no_cen_index)]]

    no_cen = no_cen[0]
    nocen_fea = nocen_fea[0]
    no_cen_headers = no_cen_headers[0]
    no_cen_headers = no_cen_headers.reshape(((len(no_cen_headers), 1)))  # (3,1)



    num = 0
    for train_index, test_index in kf.split(censor_fea):
        train_X1 = censor_fea[train_index]
        train_Y1 = censor[train_index]
        train_headers1 = censor_headers[train_index]
        test_X1 = censor_fea[test_index]
        test_Y1 = censor[test_index]
        test_headers1 = censor_headers[test_index]
        if num == fold_num:
            break
        num +=1

    num = 0
    for train_index, test_index in kf.split(nocen_fea):
        train_X2 = nocen_fea[train_index]
        train_Y2 = no_cen[train_index]
        train_headers2 = no_cen_headers[train_index]
        test_X2 = nocen_fea[test_index]
        test_Y2 = no_cen[test_index]
        test_headers2 = no_cen_headers[test_index]
        if num == fold_num:
            break
        num +=1

    train_X = np.vstack((train_X1, train_X2))
    train_Y = np.vstack((train_Y1, train_Y2))
    train_headers = np.vstack((train_headers1,train_headers2))
    test_X = np.vstack((test_X1, test_X2))
    test_Y = np.vstack((test_Y1, test_Y2))
    test_headers = np.vstack((test_headers1,test_headers2))
    return train_X, train_Y, test_X, test_Y,train_headers, test_headers

# def split_data(spilt_seed, fea, label,headers, nfold=5, fold_num=0):
#     kf = KFold(n_splits=nfold, shuffle=True, random_state=spilt_seed)
#     train_X = []
#     train_Y = []
#     test_X = []
#     test_Y = []
#     num = 0
#     for train_index, test_index in kf.split(fea):
#         train_X = fea[train_index]
#         train_Y = label[train_index]
#         train_headers = headers[train_index]
#         test_X = fea[test_index]
#         test_Y = label[test_index]
#         test_headers=headers[test_index]
#         if num == fold_num:
#             print(num)
#             break
#         num += 1
#     train_X, train_Y, train_headers = sort_surv_data(train_X, train_Y, train_headers)
#     test_X, test_Y, test_headers= sort_surv_data(test_X, test_Y, test_headers)
#     return train_X, train_Y, test_X, test_Y,train_headers, test_headers