import pandas as pd
import numpy as np
import csv
from sklearn.cluster import KMeans
from DeepRandomCox import train_CoxKmeans_pathway
from DeepRandomCox import predictCoxKmeans_pathway

import torch
from support import get_omic_data
from support import sort_data
from support import mkdir
from support import plot_curve
from support import cal_pval
from idle_gpu import idle_gpu

import torch.nn as nn
from DeepRandomCox import CoxKmeans_pathway
from captum.attr import IntegratedGradients
import warnings

warnings.filterwarnings("ignore")
### 指定使用的GPU

# gpu_id = idle_gpu()
device = torch.device("cuda:{}".format(idle_gpu()) if torch.cuda.is_available() else "cpu")
print(device)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)


### 是否画图
is_plot = False
## is drawing
nn_config = {
    "learning_rate": 0.0000007, #0.0000007
    "learning_rate_decay": 0.999,
    "activation": 'relu',
    "epoch_num": 1000,
    "skip_num": 1,
    "L1_reg": 1e-5,
    "L2_reg": 1e-5,
    "optimizer": 'Adam',
    "dropout": 0.0,
    "hidden_layers": [1000, 500, 24, 1],
    "standardize":True,
    "batchnorm":False,
    "momentum": 0.9,
    "n_clusters" : 2,
    "update_interval" : 1,
    "kl_rate":10,
    "ae_rate":1,
    "seed": 1
}



# if is_plot:
#     plot_curve(train_curve, title="train epoch-Cindex curve", x_label="epoch", y_label="Cindex")
#     plot_curve(test_curve, title="test epoch-Cindex curve", x_label="epoch", y_label="Cindex")

dataset = ["my_dataset/brca_path_batch"]
pathway_mask_path = "my_dataset/pathway_mask.csv"
# ### 5-independent TEST
train_save_set = []
valid_save_set = []
test_pred_set = []
test_y_set = []
mse_set = []
ss_set = []
classify_set = []
x_reconstuct_set = []

hidden_l = [100, 50, 20]
lr_set = [1E-7,5E-7,1E-6]
#
# hidden_l = [20]  #20,
# lr_set = [5E-7,1E-6]  #5E-7,1E-6
def get_attr(model, input_x, y_pred):
    ig = IntegratedGradients(model)
    baseline_x = torch.zeros(input_x.shape[0], input_x.shape[1]).to(device)
    baseline_y = torch.zeros(y_pred.shape[0]).to(device)
    attributions, delta = ig.attribute((input_x, y_pred), (baseline_x, baseline_y), target=0, return_convergence_delta=True)
    print('IG Attributions:', attributions)
    print('Convergence Delta:', delta)
    return attributions, delta

for filename in dataset:
    for h in hidden_l:
        for lr in lr_set:
            valid_save_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr) + filename])
            mse_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr) + filename])
            ss_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr) + filename])
            classify_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr) + filename])
            x_reconstuct_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr) + filename])
            for seed in range(1):
                for fold_num in range(10):
                    ori_train_X, ori_train_Y, ori_test_X, ori_test_Y = get_omic_data(fea_filename=(filename + ".csv"), seed=seed, nfold = 10 ,fold_num=fold_num)
                    ori_idx, ori_train_X, ori_train_Y = sort_data(ori_train_X, ori_train_Y)
                    input_nodes = len(ori_train_X[0])
                    nn_config["learning_rate"] = lr
                    nn_config["hidden_layers"][2] = h
                    pathway_mask = pd.read_csv(pathway_mask_path, index_col = 0).iloc[:,:].values
                    pathway_mask = torch.from_numpy(pathway_mask).type(torch.FloatTensor)
                    pathway_mask = pathway_mask.to(device)
                    Pathway_Nodes = int(pathway_mask.shape[0])
                    train_curve, valid_curve, model = train_CoxKmeans_pathway(device, nn_config, input_nodes, Pathway_Nodes, pathway_mask, ori_train_X, ori_train_Y, ori_test_X, ori_test_Y)
                    valid_save_set.append(valid_curve[1])
                    mse_set.append(valid_curve[2])
                    ss_set.append(valid_curve[3])
                    classify_set.append(valid_curve[4])
                    x_reconstuct_set.append(valid_curve[5])
                    if is_plot:
                        plot_curve(curve_data=train_curve, title="train epoch-Cindex curve")
                        plot_curve(curve_data=valid_curve, title="test epoch-Cindex curve")

                    test_x_bar, test_q, prediction = predictCoxKmeans_pathway(model, device, nn_config, ori_test_X)
                    test_pred_set.append(prediction)
                    test_y_set.append(ori_test_Y)
                    input_x = ori_test_X.astype(np.float32)
                    input_x = torch.from_numpy(input_x).to(device)
                    x_1 = model.tanh(model.sc1(input_x))
                    x_bar, hidden = model.ae(x_1)
                    kmeans = KMeans(n_clusters=nn_config["n_clusters"], n_init=20)
                    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
                    y_pred = torch.from_numpy(y_pred).to(device)
                    attributions, delta = get_attr(model, input_x, y_pred)

                    save_path = './result4/IG_fold'+str(fold_num)+'.csv'
                    np.savetxt(save_path, attributions[0].cpu().numpy(), delimiter=',')

            mkdir("result3")
            with open("result3/" + "Cindex" + ".csv", "w",
                      newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(valid_save_set)
            with open("result3/" + "prediction" + ".csv", "w",
                      newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(test_pred_set)
            with open("result3/" + "test_Y" + ".csv", "w",
                      newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(test_y_set)
            with open("result3/" + "testmse" + ".csv", "w",
                      newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(mse_set)
            with open("result3/" + "SS" + ".csv", "w",
                      newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(ss_set)
            with open("result3/" + "classify" + ".csv", "w",
                      newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(classify_set)
            with open("result3/" + "reconsturct" + ".csv", "w",
                      newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(x_reconstuct_set)

                