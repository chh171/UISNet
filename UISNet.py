import pandas as pd
import numpy as np
import csv
from sklearn.cluster import KMeans
from DeepRandomCox import train_CoxKmeans_pathway
from DeepRandomCox import predictCoxKmeans_pathway
from DeepRandomCox import getCindex
from DeepRandomCox import CoxKmeans_pathway
from captum.attr import IntegratedGradients

import torch
from support import load_omic_data
from support import sort_data
from support import mkdir
from support import plot_curve
from support import cal_pval
from idle_gpu import idle_gpu
import warnings

warnings.filterwarnings("ignore")


# gpu_id = idle_gpu()
device = torch.device("cuda:{}".format(idle_gpu()) if torch.cuda.is_available() else "cpu")
print(device)


is_save = True

is_load = True



is_plot = False

nn_config = {
    "learning_rate": 0.0000001,
    "learning_rate_decay": 0.999,
    "activation": 'relu',
    "epoch_num": 50,
    "skip_num": 50,
    "L1_reg": 1e-5,
    "L2_reg": 1e-5,
    "optimizer": 'Adam',
    "dropout": 0.5,
    "hidden_layers": [500, 200, 24, 1],
    "standardize":False,
    "batchnorm":False,
    "momentum": 0.9,
    "n_clusters" : 2,
    "update_interval" : 1,
    "kl_rate":0,
    "ae_rate":1,
    "seed": 1
}




dataset = ["my_dataset/brca_test"]
pathway_mask_path = "my_dataset/pathway_mask.csv"


train_save_set = []
train_pred_set = []

mse_set = []
ss_set = []
classify_set = []
x_reconstuct_set = []

hidden_l = [50]
lr_set = [1E-7]
iter_num = 5

def get_attr(model, input_x, y_pred):
    ig = IntegratedGradients(model)
    baseline_x = torch.zeros(input_x.shape[0], input_x.shape[1]).to(device)
    baseline_y = torch.zeros(y_pred.shape[0]).to(device)
    attributions, delta = ig.attribute((input_x, y_pred), (baseline_x, baseline_y), target=0, return_convergence_delta=True)
    print('IG Attributions:', attributions)
    print('Convergence Delta:', delta)
    return attributions, delta


def get_uncertainty(model, input_x, iter_num):
    attribution_list = []
    delta_list = []
    rank_list = []
    for a in range(iter_num):
        model.eval()
        x_1 = model.tanh(model.sc1(input_x))
        x_bar, hidden = model.ae(x_1)
        kmeans = KMeans(n_clusters=nn_config["n_clusters"], n_init=20)
        y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
        y_pred = torch.from_numpy(y_pred).to(device)
        x_bar, z, pred = model(input_x, y_pred)

        preds = pred.cpu().view(-1).detach().numpy()
        sorted_nums = sorted(enumerate(preds), key=lambda x: x[1])
        idx = [i[0] for i in sorted_nums]
        sorted_nums = sorted(enumerate(idx), key=lambda x: x[1])
        ranks = [i[0] for i in sorted_nums]
        rank_list.append(ranks)

        model.train()
        ig = IntegratedGradients(model)
        baseline_x = torch.zeros(input_x.shape[0], input_x.shape[1]).to(device)
        baseline_y = torch.zeros(y_pred.shape[0]).to(device)
        attributions, delta = ig.attribute((input_x, y_pred), (baseline_x, baseline_y), target=0,
                                           return_convergence_delta=True)
        save_path2 = './result_all/IG_alltrain'+str(a)+'.csv'
        np.savetxt(save_path2, attributions[0].cpu().numpy(), delimiter=',')
        attribution_list.append(attributions[0].cpu().numpy())
        delta_list.append(delta.cpu().numpy())
    return np.var(np.array(attribution_list), axis=0), np.var(np.array(delta_list), axis=0), np.var(
        np.array(rank_list), axis=0)


if is_save:
    for filename in dataset:
        for h in hidden_l:
            for lr in lr_set:
                train_save_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr) + filename])
                mse_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr) + filename])
                ss_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr) + filename])
                classify_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr) + filename])
                x_reconstuct_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr) + filename])
                for seed in range(1):
                    ori_train_X, ori_train_Y, headers = load_omic_data(filename + ".csv")
                    ori_idx, ori_train_X, ori_train_Y = sort_data(ori_train_X, ori_train_Y)
                    input_nodes = len(ori_train_X[0])
                    nn_config["learning_rate"] = lr
                    nn_config["hidden_layers"][2] = h
                    pathway_mask = pd.read_csv(pathway_mask_path, index_col = 0).iloc[:,:].values
                    pathway_mask = torch.from_numpy(pathway_mask).type(torch.FloatTensor)
                    pathway_mask = pathway_mask.to(device)
                    Pathway_Nodes = int(pathway_mask.shape[0])

                    blank, train_curve, model = train_CoxKmeans_pathway(device, nn_config, input_nodes, Pathway_Nodes, pathway_mask, ori_train_X, ori_train_Y, ori_train_X, ori_train_Y)
                    train_save_set.append(train_curve[1])




                    train_x_bar, train_q, prediction = predictCoxKmeans_pathway(model, device, nn_config, ori_train_X)
                    train_pred_set.append(prediction)

                #IG
                    input_x = ori_train_X.astype(np.float32)
                    input_x = torch.from_numpy(input_x).to(device)
                    attributions_var, delta_var, rank_var = get_uncertainty(model, input_x, iter_num)
                    save_path = './result_all/IG_uncer.csv'
                    np.savetxt(save_path, attributions_var, delimiter=',')


                    mkdir("result_all")
                    with open("result_all/" + "Cindex" + ".csv", "w",
                          newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows(train_save_set)
                    with open("result_all/" + "prediction" + ".csv", "w",
                          newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows(train_pred_set)


                    if is_load:
                        indenpendent_pred_set = []
                        indenpendent_cindex_set = []
                        indenpendent_cindex = []
                        filename2 = "my_dataset/brca_test"
                        ori_train_X, ori_train_Y, headers = load_omic_data(filename2 + ".csv")
                        train_x_bar, train_q, prediction = predictCoxKmeans_pathway(model, device, nn_config, ori_train_X)
                        prediction = prediction.reshape(1, -1)
                        indenpendent_pred_set.append(prediction)
                        indenpendent_cindex.append(getCindex(ori_train_Y, prediction))
                        indenpendent_cindex_set.append(indenpendent_cindex)
                        print(indenpendent_cindex)
                        with open("result_all/" + "indenpendent_prediction" + ".csv", "w",
                              newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            for pred in prediction:
                                writer.writerow(pred)
                        with open("result_all/" + "indenpendent_Cindex" + ".csv", "w",
                              newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            train_save_set.append(indenpendent_cindex_set)
                            writer.writerows(train_save_set)



