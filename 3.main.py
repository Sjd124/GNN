import numpy as np
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns


def generate_matrix(index_file_path, weight_file_path):
    # 读取索引文件
    with open(index_file_path, 'r') as index_file:
        source_nodes = list(map(int, map(float, index_file.readline().split())))  # 读取第一行（源节点）
        target_nodes = list(map(int, map(float, index_file.readline().split())))  # 读取第二行（目标节点）
        source_nodes= source_nodes[::2]
        target_nodes = target_nodes[::2]
        print("+++++++++++++++++++")
        print(source_nodes)
        print(target_nodes)
        print("+++++++++++++++++++")

    # 读取权重文件
    # with open(weight_file_path, 'r') as weight_file:
    #     weights = [float(weight) for weight in weight_file.readline().split()]  # 读取权重
    #     print(weight)
    # 读取权重文件
    with open(weight_file_path, 'r') as weight_file:
        weights = [float(weight.strip()) for weight in weight_file.readlines()]  # 读取权重
        print(weights)
        print(len(weights))

    # 确定矩阵的大小
    num_source_nodes = len(source_nodes)
    print("+++++++++++++")
    print(num_source_nodes)
    print("+++++++++++++")
    num_target_nodes = len(target_nodes)
    print(num_target_nodes)
    print("+++++++++++++")

    # 创建零矩阵
    # matrix = np.zeros((num_source_nodes, num_target_nodes))
    matrix = np.zeros((12, 12))
# 一根CNT有多少个粒子就是多少，比如我是6个粒子
    # 填充矩阵
    for i in range(len(weights)):
        source_index = source_nodes[i]
        target_index = target_nodes[i]
        weight = weights[i]
        matrix[source_index][target_index] = weight
        matrix[target_index][source_index] = weight

    return matrix




# 生成矩阵
matrix = generate_matrix(index_file_path, weight_file_path)

# 设置打印选项以打印完整矩阵
np.set_printoptions(threshold=np.inf)

# 打印矩阵
print("Generated Matrix:")
# for row in matrix:
#     print(row)
print(matrix)


# # 绘制热力图
# plt.imshow(matrix, cmap='hot', interpolation='nearest')
# plt.colorbar()  # 添加颜色条
# plt.show()




# 自定义颜色映射，避免黑色背景
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_cmap',
    ['#ffffff', '#ffeda0', '#feb24c', '#f03b20', '#bd0026']
)
# 定义自定义颜色映射
custom_cmap1 = LinearSegmentedColormap.from_list(
    'custom_cmap1',
    ['#ffffe0', '#ffeda0', '#feb24c', '#f03b20', '#bd0026']
)
# 定义自定义颜色映射
custom_cmap2 = LinearSegmentedColormap.from_list(
    'custom_cmap2',
    ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
)
# 定义自定义颜色映射
custom_cmap3 = LinearSegmentedColormap.from_list(
    'custom_cmap3',
    ['#e1f5fe', '#81d4fa', '#29b6f6', '#039be5', '#0288d1', '#0277bd', '#01579b']
)
# 定义自定义颜色映射
custom_cmap4 = LinearSegmentedColormap.from_list(
    'custom_cmap4',
    ['#00ff00', '#ff0000']
)
# 定义颜色
custom_cmap_purple = LinearSegmentedColormap.from_list(
    'custom_cmap_purple',
    ['#f5f5ff', '#e6e6fa', '#d8bfd8', '#d8b8d8', '#d890d8', '#d870d8', '#d850d8', '#d830d8', '#d810d8', '#800080']
)
custom_cmap_pink = LinearSegmentedColormap.from_list(
    'custom_cmap_pink',
    ['#fff0f5', '#ffe4e1', '#ffd7e7', '#ffbbd8', '#ff99c8', '#ff7ac8', '#ff5ac8', '#ff3ac8', '#ff1ac8', '#ff00a8']
)
# 绘制热力图
plt.figure(figsize=(14, 12))
# ax=sns.heatmap(matrix, cmap=custom_cmap1, vmin=0, vmax=1, cbar_kws={'label': 'Attention Score', 'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1]},cbar=False)
ax=sns.heatmap(matrix, cmap=custom_cmap_purple, vmin=0, vmax=1, cbar_kws={'label': 'Attention Score', 'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1]})
# ax.set_facecolor('#ffffe0')  # 淡黄色的十六进制颜色代码
# 设置颜色条标签的字体大小
ax.invert_yaxis()
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=28)  # 调整颜色条刻度标签的字体大小
cbar.set_label('Attention Score', fontsize=30)  # 调整颜色条标签的字体大小

# sns.heatmap(matrix, cmap='inferno', vmin=0, vmax=1, cbar_kws={'label': 'Attention Score', 'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1]})
# plt.imshow(matrix, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
# plt.colorbar(label='Attention Score', ticks=[0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=15)
#
# cbar = plt.colorbar(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
# cbar.set_label('Attention Score', fontsize=15)
# cbar.ax.tick_params(labelsize=12)

# 设置 x 和 y 轴标签
plt.title('Cp7_Attention Heatmap', fontsize=30)
plt.xlabel('Particle number', fontsize=25)
plt.ylabel('Particle number', fontsize=25)

number = 1
# 设置 x 和 y 轴刻度从 1 开始
num_ticks = matrix.shape[0]
print(num_ticks)
plt.xticks(ticks=np.arange(0.5, num_ticks, 1), labels=np.arange(number, num_ticks + number, 1), fontsize=38)
plt.yticks(ticks=np.arange(0.5, num_ticks, 1), labels=np.arange(number, num_ticks + number, 1), fontsize=38)
# # 隐藏y轴的坐标
# plt.gca().set_yticks([])
# 保存图像为 SVG 文件
plt.savefig("/mnt/ssd1/data1/sjd/new/cnt12/3/cp7/11.png", dpi=300)

# # 显示图像
# plt.show()