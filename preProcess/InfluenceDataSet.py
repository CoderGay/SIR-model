# If you have installed dgl-cuXX package, please uninstall it first.
# ! pip install  dgl -f https://data.dgl.ai/wheels/repo.html
# ! pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html

from utils.const import const
import torch
import random
import scipy.sparse as sp
import numpy as np
import pandas as pd
import math
import datetime
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import torch.nn.functional as F
# The PyG built-in GCNConv
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as pyg_T
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.utils import from_networkx
import torch_geometric as pyg
import torch_geometric.nn.conv as gconv

random.seed(9527)
np.random.seed(9527)
torch.random.seed()
torch.manual_seed(826)
torch.cuda.manual_seed(826)
device = 'cpu'  # change to 'cuda' for GPU
pyg.seed_everything(9527)


class InfluenceDataSet:

    # def __init__(self, edges_data: pd.DataFrame, src_index: str = 'user_id', dst_index: str = 'friend_id',
    #              num_nodes: int = 0):
    #     self.edges_data = edges_data
    #     # 用户id与节点的id映射
    #     src, dst = self.__dict(edges_data[src_index].to_numpy(), edges_data[dst_index].to_numpy())
    #     edges_tensor = torch.tensor(np.array([src, dst]), dtype=torch.long)
    #     print(edges_tensor)
    #     # self.dgl_graph = dgl.graph((src, dst), idtype=torch.int32)
    #     # self.pyg_graph.ndata['status'] = torch.zeros(self.dgl_graph.number_of_nodes(), dtype=torch.int8)
    #     self.pyg_graph = Data(edge_index=edges_tensor, transform=pyg_T.ToSparseTensor(remove_edge_index=False))  # pyg图
    #
    #     # 初始没有节点被感染, 将所有节点的状态设置为 易感者（S）:0
    #     if num_nodes > 0 and num_nodes > len(self.id2node):
    #         self.pyg_graph.num_nodes = num_nodes
    #     else:
    #         self.pyg_graph.num_nodes = len(self.id2node)
    #         if num_nodes > 0:
    #             print('UserWarning: the number of nodes that you input(%d) is too less\nThere are %d nodes indeed.' % (num_nodes, len(self.id2node)))
    #     status_matrix = torch.zeros(self.pyg_graph.num_nodes, 3, dtype=torch.int8)  # SIR矩阵[num_nodes * 3]
    #     # ones_column = torch.ones(self.pyg_graph.num_nodes, 1)
    #     # status_matrix[:, 0] = ones_column[:, 0]   # 将第0列赋值为1
    #     self.pyg_graph.x = status_matrix
    #     self.pyg_graph.transform(self.pyg_graph)  # <- 将edge_index转换为稀疏矩阵
    def __init__(self, data_source, num_nodes: int = 0):

        if isinstance(data_source, pd.DataFrame):
            data = self.from_pd(data_source)
            if num_nodes > 0 and num_nodes > len(self.id2node):
                data.num_nodes = num_nodes
            else:
                data.num_nodes = len(self.id2node)
                if num_nodes > 0:
                    print(
                        'UserWarning: the number of nodes that you input(%d) is too less\nThere are %d nodes indeed.' % (
                        num_nodes, len(self.id2node)))
        else:
            data = self.from_netwokrx(data_source)

        # 初始没有节点被感染, 将所有节点的状态设置为 易感者（S）:0
        print(data)
        status_matrix = torch.zeros(data.num_nodes, 3, dtype=torch.int8)  # SIR矩阵[num_nodes * 3]
        # ones_column = torch.ones(self.pyg_graph.num_nodes, 1)
        # status_matrix[:, 0] = ones_column[:, 0]   # 将第0列赋值为1
        data.x = status_matrix
        data.transform(data)
        self.pyg_graph = data

    def from_netwokrx(self, nx_graph):
        data = from_networkx(nx_graph)
        data.transform = pyg_T.ToSparseTensor(remove_edge_index=False)
        return data

    def from_pd(self, edges_data: pd.DataFrame, src_index: str = 'user_id', dst_index: str = 'friend_id'):
        self.edges_data = edges_data
        # 用户id与节点的id映射
        src, dst = self.__dict(edges_data[src_index].to_numpy(), edges_data[dst_index].to_numpy())
        edges_tensor = torch.tensor(np.array([src, dst]), dtype=torch.long)
        print(edges_tensor)
        # self.dgl_graph = dgl.graph((src, dst), idtype=torch.int32)
        # self.pyg_graph.ndata['status'] = torch.zeros(self.dgl_graph.number_of_nodes(), dtype=torch.int8)
        transform = pyg_T.ToSparseTensor(remove_edge_index=False)
        data = Data(edge_index=edges_tensor, transform=transform)  # pyg图
        return data

    def __dict(self, arr_1: np.ndarray, arr_2: np.ndarray) -> dict:
        # TODO
        """
        将数据集的节点id转化为pyg的node_index，通过dict {id: node_index}映射
        :param self.node_list
        :return: src_node_index: np.ndarray, dst_node_index: np.ndarray
        """
        node_list = np.unique(np.concatenate((arr_1, arr_2)))
        self.number_of_node = len(node_list)
        self.id2node = {}  # {id: node_index}
        for node_index, src_id in enumerate(node_list):
            self.id2node[src_id] = node_index
        src_node_index = np.array(list(map(self.id2node.get, arr_1)))
        dst_node_index = np.array(list(map(self.id2node.get, arr_2)))
        return src_node_index, dst_node_index

    def visualize_embedding(self, color, epoch=None, loss=None):
        plt.figure(figsize=(7, 7))
        plt.xticks([])
        plt.yticks([])
        h = self.embedding  # 请初始化embedding
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
        plt.show()

    def visualize_graph(self, color):
        # G = to_networkx(data, to_undirected=True)
        '''
        :param color: data.y
        :return: none
        '''
        plt.figure(figsize=(7, 7))
        plt.xticks([])
        plt.yticks([])
        nx_graph = to_networkx(self.pyg_graph, to_undirected=True)
        nx.draw_networkx(nx_graph, pos=nx.spring_layout(nx_graph, seed=9527), with_labels=True,
                         node_color=color, cmap="Set2")
        plt.show()

    def visualize_nx(self):
        # nx.draw(self.dgl_graph.to_networkx(), with_labels=True)
        nx_graph = nx.from_edgelist(self.pyg_graph.edge_index.numpy().T)
        nx.draw(nx_graph, with_labels=True)
        plt.show()  # You should cancel this clause if run in colab and import %matplotlib inline

    def get_node_status(self):
        """
            计算当前图内各个状态节点的数目
            :param self: 输入自身存储的dgl图
            :return: 各个状态（S、I、R）的节点数目
        """
        s_num, i_num, r_num = 0, 0, 0
        # 统计各个状态（S、I、R）的节点数目

        # s_num, i_num, r_num = self.dgl_graph.ndata['status'].bincount().tolist()
        # static_list = self.pyg_graph.x.bincount().tolist()
        # s_num, i_num, r_num = static_list + [0] * (3 - len(static_list))
        s_num, i_num, r_num = self.pyg_graph.x.sum(dim=0).tolist()
        return s_num, i_num, r_num

    def __pyg_get_neighbors(self, node: int, directed: str = 'undirected') -> torch.tensor:
        """
        获取pyg的邻居节点,私有函数
        :param node: 节点序数
        :param directed: 默认无向图, predecessors找前继节点，successors找后续节点
        :return: tensor
        """
        edge_index = self.pyg_graph.edge_index
        src, dst = edge_index
        neighbors = dst[src == node]  # 默认找后续节点,即node指向的邻居节点
        if directed == 'predecessors':  # 找前继节点,即指向node的邻居节点
            neighbors = src[dst == node]
        if directed == 'undirected':  # 无向图
            neighbors = torch.cat((neighbors, src[dst == node]), dim=0)  # 添加前继节点,即指向node的邻居节点
            neighbors = torch.unique(neighbors, return_inverse=False, return_counts=False)
        return neighbors

    def update_node_status(self, beta, gamma):
        """
        更新节点状态
        :param self.graph: 网络图
        :param node: 节点序数
        :param beta: 感染率
        :param gamma: 免疫率
        """
        # TODO
        # 如果当前节点状态为 感染者(I) 有概率gamma变为 免疫者(R)
        p_R = torch.rand(self.pyg_graph.num_nodes, 1)
        ones_column = torch.ones(self.pyg_graph.num_nodes, 1)  # 创建一个大小为(n, 1)且填充为1的tensor
        # zeros_column = torch.zeros(self.pyg_graph.num_nodes, 1)  # 创建一个大小为(n, 1)且填充为0的tensor
        p_R = torch.cat((ones_column, p_R, ones_column), dim=1)
        self.pyg_graph.x = p_R * self.pyg_graph.x
        i_old_tensor = self.pyg_graph.x[:, 1]
        condition_r = (i_old_tensor > 0) & (i_old_tensor < gamma)
        r_new_tensor = torch.where(condition_r, 1, 0)
        i_new_tensor = torch.where(condition_r, 0, i_old_tensor)
        self.pyg_graph.x[:, 2] = self.pyg_graph.x[:, 2] + r_new_tensor
        self.pyg_graph.x[:, 1] = i_new_tensor
        condition_i_r = (i_new_tensor > 0) & (i_new_tensor < 1)
        i_new_tensor = torch.where(condition_i_r, 1, i_new_tensor)
        self.pyg_graph.x[:, 1] = i_new_tensor
        self.pyg_graph.x = self.pyg_graph.x.type(torch.int8)    # 计算时自动转化为float了, 得把类型转换回来

        # TODO
        # 如果当前节点状态为 易感染者(S) 有概率beta变为 感染者(I)

        S_tensor = self.pyg_graph.x[:, 0]
        S_tensor = S_tensor.reshape(self.pyg_graph.num_nodes, 1)
        # S_matrix = S_tensor.expand(self.pyg_graph.num_nodes, self.pyg_graph.num_nodes)

        sparse_tensor = self.pyg_graph.adj_t.detach().to_torch_sparse_coo_tensor()

        S_matrix = sparse_tensor * S_tensor

        row = S_matrix._indices()[0]
        col = S_matrix._indices()[1]
        data = S_matrix._values()
        shape = S_matrix.size()
        S_coo_matrix = sp.coo_matrix((data, (row, col)), shape=shape)
        rows, cols = S_coo_matrix.sum(axis=1).nonzero()
        # 生成num_nonzero_raw行概率矩阵P num_nodes * num_nodes 但是要稀疏矩阵
        # 随机生成(非零行 x num_nodes) 的非零元素
        values = torch.rand(len(rows) * self.pyg_graph.num_nodes)
        # 指定非零元素所在的行号
        row_indices = torch.tensor(rows)
        # 创建列号的范围 [0, 1, 2,..., num_nodes]
        col_indices = torch.arange(self.pyg_graph.num_nodes)
        # 将行号和列号组合成 indices
        indices = torch.stack([row_indices.repeat_interleave(self.pyg_graph.num_nodes), col_indices.repeat(len(rows))])
        # 创建稀疏矩阵
        p_r_coo_matrix = torch.sparse_coo_tensor(indices, values, size=[self.pyg_graph.num_nodes, self.pyg_graph.num_nodes])

        i_matrix = S_matrix * p_r_coo_matrix

        nonzero_i_coo_tensor = i_matrix.index_select(dim=0, index=row_indices)
        nonzero_i_tensor = nonzero_i_coo_tensor.to_dense()

        ones_row = torch.ones(nonzero_i_tensor.shape[0], self.pyg_graph.num_nodes)  # nonzero_rows * num_nodes
        zeros_row = torch.zeros(nonzero_i_tensor.shape[0], self.pyg_graph.num_nodes)

        i_condition = (nonzero_i_tensor > 0) & (nonzero_i_tensor < beta)
        nonzero_i_tensor = torch.where(i_condition, ones_row, zeros_row)
        i_status_index = torch.where(nonzero_i_tensor.sum(dim=1) > 0, row_indices, -1)

        i_status_index = i_status_index[i_status_index >= 0]

        self.pyg_graph.x[i_status_index, 1] = 1
        self.pyg_graph.x[i_status_index, 0] = 0

    def __init_x(self):
        self.pyg_graph.x = torch.zeros(self.pyg_graph.num_nodes, 3, dtype=torch.int8)
        self.pyg_graph.x[:, 0] = torch.ones(self.pyg_graph.num_nodes, dtype=torch.int8)

    def __set_i_seeds(self, seed_set: list):
        self.pyg_graph.x[seed_set, 0] = 0
        self.pyg_graph.x[seed_set, 1] = 1

    def SIR_network(self, source, beta, gamma, step):
        """
        获得感染源的节点序列的SIR感染情况
        :param graph: networkx创建的网络
        :param source: 需要被设置为感染源的节点Id所构成的序列
        :param beta: 感染率
        :param gamma: 免疫率
        :param step: 迭代次数
        """
        n = self.pyg_graph.num_nodes  # 网络节点个数
        sir_values = []  # 存储每一次迭代后网络中感染节点数I+免疫节点数R的总和
        # 初始化节点状态
        # 将所有节点的状态设置为 易感者（S）
        self.__init_x()
        # 设置初始感染源
        # 将感染源序列中的节点设置为感染源，状态设置为 感染者（I）
        self.__set_i_seeds(source)
        # 记录初始状态
        sir_values.append(len(source) / n)
        # 开始迭代感染
        for s in range(step):
            # 针对对每个节点进行状态更新以完成本次迭代
            # if gamma >= 1:  # 如果gamma=1，开启上一轮迭代的感染节点全部会变免疫的模式
            #     last_I_nodes = []
            #     for node in range(n):  # 记录这轮迭代未开始时的被感染节点I
            #         if self.pyg_graph.x[node] == const.I:
            #             last_I_nodes.append(node)

            self.update_node_status(beta, gamma)  # 针对node号节点进行SIR过程
            # print('已迭代%d轮' % step)
            s, i, r = self.get_node_status()  # 得到本次迭代结束后各个状态（S、I、R）的节点数目
            sir = (i + r) / n  # 该节点的sir值为迭代结束后 感染节点数i+免疫节点数r
            sir_values.append(sir)  # 将本次迭代的sir值加入数组
            # if gamma >= 1:  # 如果gamma=1，开启上一轮迭代的感染节点全部会变免疫的模式
            #     for node in last_I_nodes:
            #         self.pyg_graph.x[node] = const.R  # 将上一轮感染源序列中的节点设置为恢复者，状态设置为 恢复者（R）

        return sir_values


if __name__ == '__main__':
    # spmat = sp.rand(100, 100, density=0.05)  # 5%非零项
    # g_2 = dgl.from_scipy(spmat)
    # nx.draw(g_2.to_networkx(), with_labels=True)
    # plt.show()

    # 先设置header=None，否则会自动把第一列数据作为列名
    edges_pd_data = pd.read_csv('../data/raw_data/digg/digg_friends.csv', header=None)
    # 重新设置列名columns
    edges_pd_data.columns = ['mutual', 'friend_date', 'user_id', 'friend_id']
    # 对其中一个列名重新设置
    # edges_data.rename(columns={'data':'DATA'})
    print(edges_pd_data)
    # number_of_nodes = 139409  # the number of node in friendship is 71,367, but voted by 139,409 users
    number_of_nodes = 115
    nx_g_karate = nx.read_gml(r'../data/raw_data/football/football.gml', label='id')
    diggDataSet = InfluenceDataSet(nx_g_karate, number_of_nodes)
    # diggDataSet = InfluenceDataSet(edges_pd_data, 'user_id', 'friend_id', number_of_nodes)
    print(diggDataSet.pyg_graph)
    # diggDataSet.visualize_graph(diggDataSet.pyg_graph.x)

    '''
        SIR传播
        '''
    dfSIR = pd.DataFrame(columns=['Id', 'SIR'])
    str_fm = "{0:^5}\t{1:^10}"  # 格式化输出
    print(str_fm.format("Id", "SIR"))
    # 循环所有节点
    for j in range(number_of_nodes):
        node_id = j + 1  # 索引序列[0~n-1]，此处+1转回为Id
        # 由于SIR为概率模型，我们进行多次实验取平均值，实验次数可自行设置
        n = 100  # 实验次数
        sir_list = []
        for k in range(n):
            # SIR参数设置，可自行设置
            beta = 0.001  # 感染率 0.5
            gamma = 1  # 免疫率 0.1
            step = 100  # SIR模型中的感染传播轮次
            # 节点的感染情况
            sir_source = [j]  # 方法输入为数组，将节点强制转换为数组，且SIR实现中使用的为节点索引号[0~n-1]，此处使用j索引号
            sir_values = diggDataSet.SIR_network(sir_source, beta, gamma, step)
            # Fc = sir_values[step - 1]  # 最终的感染范围
            Fc = sir_values[step]
            # 由于有概率出现节点直接免疫，传播停止的“异常”情况
            # 我们设置阈值，只统计传播覆盖范围大于1%（0.01）的情况
            if Fc > 0.01:
                sir_list.append(Fc)
        if sir_list is None or len(sir_list) == 0:
            sir = -1
        else:
            sir = np.mean(sir_list)  # 对100实验的输出结果求均值
        print(str_fm.format(node_id, sir, chr(12288)))
        # 添加至dataframe
        dfSIR = dfSIR.append({
            'Id': int(node_id),
            'SIR': sir
        }, ignore_index=True)
    '''
    输出到文件。更换为自己的数据文件！！！
    '''
    dfSIR.to_csv('../result/Node-SIR.csv', index=False)
