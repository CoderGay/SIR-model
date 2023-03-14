import random
import numpy as np
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd

'''
程序主要功能
输入：网络图邻接矩阵，需要被设置为感染源的节点序列，感染率，免疫率，迭代次数step
输出：被设置为感染源的节点序列的SIR感染情况---每次的迭代结果（I+R）/n
'''
random.seed(9527)
np.random.seed(9527)


def update_node_status(graph, node, beta, gamma):
    """
    更新节点状态
    :param graph: 网络图
    :param node: 节点序数
    :param beta: 感染率
    :param gamma: 免疫率
    """

    if gamma < 1:
        # 如果当前节点状态为 感染者(I) 有概率gamma变为 免疫者(R)
        if graph.nodes[node]['status'] == 'I':
            p = random.random()
            if p < gamma:
                graph.nodes[node]['status'] = 'R'
    # 如果当前节点状态为 易感染者(S) 有概率beta变为 感染者(I)
    if graph.nodes[node]['status'] == 'S':
        # 获取当前节点的邻居节点
        # 无向图：G.neighbors(node)
        # 有向图：G.predecessors(node)，前驱邻居节点，即指向该节点的节点；G.successors(node)，后继邻居节点，即该节点指向的节点。
        # neighbors = list(graph.predecessors(node))
        neighbors = list(graph.neighbors(node))
        # 对当前节点的邻居节点进行遍历
        for neighbor in neighbors:
            # 邻居节点中存在 感染者(I)，则该节点有概率被感染为 感染者(I)
            if graph.nodes[neighbor]['status'] == 'I':
                p = random.random()
                if p < beta:
                    graph.nodes[node]['status'] = 'I'
                    break


def count_node(graph):
    """
    计算当前图内各个状态节点的数目
    :param graph: 输入图
    :return: 各个状态（S、I、R）的节点数目
    """
    s_num, i_num, r_num = 0, 0, 0
    for node in graph:
        if graph.nodes[node]['status'] == 'S':
            s_num += 1
        elif graph.nodes[node]['status'] == 'I':
            i_num += 1
        else:
            r_num += 1
    return s_num, i_num, r_num


def SIR_network(graph, source, beta, gamma, step):
    """
    获得感染源的节点序列的SIR感染情况
    :param graph: networkx创建的网络
    :param source: 需要被设置为感染源的节点Id所构成的序列
    :param beta: 感染率
    :param gamma: 免疫率
    :param step: 迭代次数
    """
    n = graph.number_of_nodes()  # 网络节点个数
    sir_values = []  # 存储每一次迭代后网络中感染节点数I+免疫节点数R的总和
    # 初始化节点状态
    for node in graph:
        graph.nodes[node]['status'] = 'S'  # 将所有节点的状态设置为 易感者（S）
    # 设置初始感染源
    for node in source:
        graph.nodes[node]['status'] = 'I'  # 将感染源序列中的节点设置为感染源，状态设置为 感染者（I）
    # 记录初始状态
    sir_values.append(len(source) / n)
    # 开始迭代感染
    for s in range(step):
        # 针对对每个节点进行状态更新以完成本次迭代
        if gamma >= 1:  # 如果gamma=1，开启上一轮迭代的感染节点全部会变免疫的模式
            last_I_nodes = []
            for node in graph:  # 记录这轮迭代未开始时的被感染节点I
                if graph.nodes[node]['status'] == 'I':
                    last_I_nodes.append(node)
        for node in graph:
            update_node_status(graph, node, beta, gamma)  # 针对node号节点进行SIR过程
        s, i, r = count_node(graph)  # 得到本次迭代结束后各个状态（S、I、R）的节点数目
        sir = (i + r) / n  # 该节点的sir值为迭代结束后 感染节点数i+免疫节点数r
        sir_values.append(sir)  # 将本次迭代的sir值加入数组
        if gamma >= 1:  # 如果gamma=1，开启上一轮迭代的感染节点全部会变免疫的模式
            for node in last_I_nodes:
                graph.nodes[node]['status'] = 'R'   # 将上一轮感染源序列中的节点设置为恢复者，状态设置为 恢复者（R）

    return sir_values


if __name__ == '__main__':
    '''
    数据准备。更换为自己的数据文件！！！
    '''
    spmat = sp.rand(100, 100, density=0.05, random_state=9527)  # 5%非零项,随机生成100*100的稀疏矩阵
    # g_2 = dgl.from_scipy(spmat)

    # adj = np.loadtxt('data/adj.txt', dtype=np.int)  # 网络的邻接矩阵
    # graph = nx.from_numpy_matrix(adj)  # 网络图：默认无向图；nx.DiGraph(adj)为创建有向图
    graph = nx.from_scipy_sparse_matrix(spmat)  # 网络图：默认无向图；nx.DiGraph(adj)为创建有向图

    nodes_n = graph.number_of_nodes()  # 节点数
    print('共 ' + str(nodes_n) + ' 个节点！')
    nx.draw(graph, with_labels=True)
    plt.show()

    '''
    SIR传播
    '''
    dfSIR = pd.DataFrame(columns=['Id', 'SIR'])
    str_fm = "{0:^5}\t{1:^10}"  # 格式化输出
    print(str_fm.format("Id", "SIR"))
    # 循环所有节点
    for j in range(nodes_n):
        node_id = j + 1  # 索引序列[0~n-1]，此处+1转回为Id
        # 由于SIR为概率模型，我们进行多次实验取平均值，实验次数可自行设置
        n = 100  # 实验次数
        sir_list = []
        for k in range(n):
            # SIR参数设置，可自行设置
            beta = 0.1  # 感染率 0.5
            gamma = 1  # 免疫率 0.1
            step = 100  # SIR模型中的感染传播轮次
            # 节点的感染情况
            sir_source = [j]  # 方法输入为数组，将节点强制转换为数组，且SIR实现中使用的为节点索引号[0~n-1]，此处使用j索引号
            sir_values = SIR_network(graph, sir_source, beta, gamma, step)
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
    dfSIR.to_csv('./result/Node-SIR.csv', index=False)
