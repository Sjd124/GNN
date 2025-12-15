import os.path as osp
import pandas as pd
import torch
import torch.nn.functional as F
import sklearn
from torch_geometric.datasets import Planetoid, Amazon, PPI, Reddit, Coauthor, CoraFull, gnn_benchmark_dataset, Flickr, \
    CitationFull, Amazon, Actor, CoraFull
from torch_geometric.data import NeighborSampler
# from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
# from torch_geometric.nn import GATv2Conv, TopKPooling, PairNorm
from torch_geometric.nn import global_mean_pool, global_max_pool

from sklearn.model_selection import StratifiedKFold
##########################################################3
# from ogb.nodeproppred import PygNodePropPredDataset
############################################################
import torch_geometric.transforms as T
import random
import numpy as np
# from torch_sparse import SparseTensor, coalesce
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import networkx as nx
import matplotlib.pyplot as plt
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

time_start = time.time()  # 记录开始时间

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == "__main__":
    dataset = MyOwnDataset()
    dataset = dataset.process()

    length_data = dataset.len()
    idx = random.randint(0, length_data)

    print(length_data, idx)

    data = dataset.get(idx=idx)

    # G = to_networkx(data, to_undirected=False)
    # visualize_graph(G, color=data.y)

    model = GCN()
    print(model)
    # model = model.to(device)

    out, h = model(data.x, data.edge_index, torch.tensor([0]))

    print('out.shape:', out.shape)
    print(f'Embedding shape: {list(h.shape)}')

    # visualize_embedding(h, color=data.y)

    # criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Define optimizer.#它是一种自适应学习率优化算法


    def train(data):
        optimizer.zero_grad()  # Clear gradients.清除模型的梯度
        #进行反向传播之前，需要清除之前的梯度，否则梯度会累积，导致不正确的更新。
        out, h = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        #data.x：是图中节点的特征矩阵，形状通常是 (num_nodes, feature_dim)。
         #data.edge_index：是图的边索引，通常表示图中边的连接关系。
        out = out.squeeze(-1)
        # print(out)
        loss = criterion(out[data.train_mask],
                         data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        #这行代码计算模型预测值和真实标签之间的损失
        # print("===============")
        # print(loss)
        # print("===============")
        #data.y[data.train_mask]：这是训练集的真实标签。通过 train_mask 提取训练节点的真实标签。
        #out[data.train_mask]：这是模型在训练节点上的预测输出。
        #train_mask 是一个掩码，它标记了哪些节点是训练集的一部分，而其他节点（例如验证集或测试集）不会参与损失计算。这样可以确保训练只发生在训练数据上。
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        #这行代码使用计算出来的梯度更新模型的参数。优化器（如 Adam）会根据学习率和梯度来调整模型的权重
        return loss, h


    def val(data):#评估模式
        model.eval()
        # print("+++++++++++++++++++++")
        # print(data.batch)
        # print("+++++++++++++++++++++")
        out1, h1 = model(data.x, data.edge_index, data.batch)
        # Perform a single forward pass.
        predict1 = out1[data.val_mask]

        out2, h2 = model(data.x, data.edge_index, data.batch)

        predict2 = out2[data.val_mask]
        # Compute the loss solely based on the training nodes.
        # print(predict1)
        # print(predict2)

        predict = (predict1 + predict2) / 2

        test_y = data.y[data.val_mask]
        # print("+++++++++++++")
        # print(predict)
        # print(mae_score)
        # print(rmse_score)
        # print("+++++++++++++")

        # predict_np = predict.detach().cpu().numpy().reshape(-1)
        # test_y_np = test_y.detach().cpu().numpy().reshape(-1)
        # print("+++++++++++++++++++++++++")
        # print(test_y.size())
        # print(predict.size())
        # print(test_y)
        # print(predict)
        # print(predict.detach())
        # print(predict.detach().numpy())
        # print("+++++++++++++++++++++++++")
        mae_score = mean_absolute_error(test_y, predict.detach().numpy())
        rmse_score = mean_squared_error(test_y, predict.detach().numpy()) ** 0.5
        r2 = r2_score(test_y, predict.detach())
        # print("+++++++++++++")
        # # print(predict)
        # print(mae_score)
        # print(rmse_score)
        # print("+++++++++++++")


        # df = pd.DataFrame({
        #     'true_values': test_y_np,
        #     'predictions': predict_np
        # })
        # df.to_csv('GCN_pro/cp5_1/validation_predictions.csv', index=False)
        return mae_score, rmse_score, r2

#训练和验证流程
    train_data_scope = length_data * 0.8


    print(int(train_data_scope), length_data)

    BATCH_SIZE = 8

    train_loader = DataLoader(dataset[:int(train_data_scope)], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset[int(train_data_scope):length_data], batch_size=BATCH_SIZE, shuffle=True)

    print(len(train_loader), len(val_loader))
    # for data in train_loader:
    #     print(data)
    #     print("++++++++++++++++++")

   

    for epoch in range(100):
        for batch in train_loader:
            loss, h = train(batch)
    torch.save(model, '/mnt/ssd2/data1/sjd/S6B10S6CNT6/new/3/cp8/1/model_gcn_pro.pt')#当前的生成模型
#验证过程
    for epoch in range(10):
        for batch in val_loader:
            mae_score, rmse_score, r2 = val(batch)

        print('epoch_val %d | mae_score: %.8f | rmse_score: %.8f | r2: %.8f:' % (epoch, mae_score, rmse_score, r2))
        time.sleep(0.3)

time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒
print(time_sum)

