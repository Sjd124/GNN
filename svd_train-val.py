import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.lines as mlines

# 修改部分：处理node_1.csv到node_1001.csv文件
num = 7

def process_file(file_path):
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 跳过第一行和最后一行
    data_lines = lines[1:-1]

    # 初始化一个列表来存储矩阵的行
    matrix = []

    # 处理每一行
    for line in data_lines:
        # 按逗号分隔（CSV文件格式）
        columns = line.strip().split(',')

        # 提取第4、5、6列（索引为3、4、5）并转换为浮点数
        row = [float(columns[3]), float(columns[4]), float(columns[5])]
        matrix.append(row)

    # 将行列表转换为NumPy数组
    matrix = np.array(matrix)

    return matrix


def process_directory(directory_path):
    # 初始化一个列表来存储每个文件的矩阵
    matrices = []

    # 遍历文件名从node_1.csv到node_1001.csv
    for i in range(1, 1002):  # 从1到1001
        file_name = f'node_{i}.csv'
        file_path = os.path.join(directory_path, file_name)

        # 检查文件是否存在
        if os.path.exists(file_path):
            matrix = process_file(file_path)
            matrices.append(matrix)
        else:
            print(f"File {file_path} does not exist. Skipping...")

    return matrices


# 示例用法
directory_path = f'/mnt/ssd1/data1/sjd/S6B10S6CNT12/SBS链/cp7'
matrices = process_directory(directory_path)

############################################################
############################################################
# 下面是降维

def svd_gai(M):
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    return s

singular_values = []

with ThreadPoolExecutor() as executor:
    singular_values = list(executor.map(svd_gai, matrices))
print(singular_values[0].shape)
print(singular_values)

# 将一维向量转换成numpy数组
singular_values = np.array(singular_values)
print(singular_values.shape)

# 数据标准化
scaler = StandardScaler()
all_singular_values = scaler.fit_transform(singular_values)

# 使用t-SNE进行降维到二维空间
tsne = TSNE(n_components=2, random_state=42)
points = tsne.fit_transform(all_singular_values)
print(points.shape)

# 打印每个矩阵转换后的点
for i, point in enumerate(points):
    print(f"Matrix {i+1}: ({point[0]}, {point[1]})")


# 随机拆分成训练集和验证集
train_points, test_points = train_test_split(points, test_size=0.2, random_state=42)

######################################################################################################


# 修改坐标轴刻度的字体大小
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)

legend = plt.legend(fontsize=27.8, loc='lower right', markerscale=1.1)
for handle in legend.legend_handles:
    handle.set_linewidth(4)

# 设置坐标轴的范围
plt.xlim(-57, 57)  # 根据你的数据范围调整
plt.ylim(-59, 59)  # 根据你的数据范围调整

# 设置边框粗细
ax = plt.gca()
ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)

# 将刻度指向图像内部
ax.tick_params(axis='both', direction='in', length=10, width=3)
plt.gca().set_facecolor('white')  # 设置背景颜色为白色
plt.tight_layout()  # 自动调整布局
plt.grid(True, linewidth=2)  # 添加网格线
plt.text(-42, 52, f'cp{num}', fontsize=55, ha='left', va='top', fontweight='bold')

# 保存高分辨率图像
plt.savefig(f"/mnt/ssd1/data1/sjd/S6B10S6CNT12/SBS链/cp7/cp7.png", dpi=400, bbox_inches='tight', pad_inches=0.1)