import networkx as nx
import matplotlib.pyplot as plt

# 读取graph文件
G = nx.read_gml(r'../data/raw_data/football/football.gml')


# 打印图的信息
print(nx.info(G))

# 对图进行可视化
nx.draw(G, with_labels=False)
plt.show()