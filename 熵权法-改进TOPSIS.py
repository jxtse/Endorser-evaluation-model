import numpy as np
import pandas as pd

# 从CSV文件中读取数据
data = pd.read_csv('data3.csv')

# 数据预处理阶段：数据清洗、数据归一化
def preprocess(data):
    # 对数据进行归一化处理
    data_norm = (data - data.min()) / (data.max() - data.min())
    # 构建指标关系矩阵
    corr_mat = np.corrcoef(data_norm, rowvar=False)
    return corr_mat

def pca(data, num_components):
    # 对数据进行中心化处理
    data_centered = data - data.mean()
    # 计算协方差矩阵
    cov_matrix = np.cov(data_centered.T)
    # 计算特征向量和特征值
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    # 按特征值大小排序
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs.sort(reverse=True)
    # 选取前num_components个特征向量构成变换矩阵
    matrix_w = np.hstack([eig_pairs[i][1].reshape(len(data.columns),1) for i in range(num_components)])
    # 将数据投影到新的空间中
    transformed_data = data_centered.dot(matrix_w)
    # 返回投影后的数据和各主成分对应的特征值
    return transformed_data, [eig_pairs[i][0] for i in range(num_components)]


# 使用熵权法计算指标权重
def entropy_weight(data):
    # 计算每个指标在各个样本中的占比
    p = data / data.sum(axis=0)
    # 计算信息熵
    entropy = (-p * np.log(p)).sum(axis=0)
    # 计算权重
    weight = (1 - entropy) / sum(1 - entropy)
    print("weight:",weight)
    return weight

# 计算正负理想解
def ideal_solution(data, weight):
    # 根据权重计算每个样本的加权得分
    data_weighted = data * weight
    # 计算正理想解和负理想解
    z_max = np.max(data_weighted, axis=0)
    z_min = np.min(data_weighted, axis=0)
    return z_max, z_min

# 使用灰色关联度法计算样本之间的距离
def grey_relation(data, z_max, z_min):
    # 计算每个样本与正负理想解之间的距离
    distance = np.zeros((data.shape[0], 2))
    for i in range(data.shape[0]):
        distance[i][0] = np.sqrt(((data.iloc[i,:] - z_max)**2).sum())
        distance[i][1] = np.sqrt(((data.iloc[i,:] - z_min)**2).sum())
    # 计算每个样本与正负理想解的关联度
    relation = distance[:,1] / (distance[:,0] + distance[:,1])
    return relation

# 计算得分和排名
def rank(data):
    weight = entropy_weight(data)
    # 计算正负理想解
    z_max, z_min = ideal_solution(data, weight)
    # 使用灰色关联度法计算每个样本与正负理想解之间的关联度
    relation = grey_relation(data, z_max, z_min)
    # 根据关联度计算每个样本的得分
    score = relation.argsort().argsort() + 1
    # 将score转化为排名
    ranking = pd.DataFrame(score, index=data.index, columns=['Rank'])
    # 根据Rank列对样本进行排名
    ranking = ranking.sort_values(by='Rank', ascending=True)
    # 输出结果
    print(ranking)
    return score

# 数据预处理
preprocess(data)

# 选择需要进行PCA分析的列
columns = ['Business Value', 'Specialty Popularity', 'Carrying Capacity', 'V Index', 'Weibo fans(w)', 'Weibo hypertalk(w)', 'Group friend']

# 进行PCA分析
pca_data, eigenvalues = pca(data[columns], 3)

# 打印前5行投影后的数据和各主成分对应的特征值
print(pca_data.head())
print(eigenvalues)

# 输出排名
rank(data)
score = rank(data)
print("score:",score)
ranking = pd.DataFrame(score, index=data.index, columns=['Rank'])
ranking.to_csv('ranking.csv', index=True)

# 将排名添加到原始数据中
result = pd.concat([data, score], axis=1)

# 保存结果
result.to_csv('result.csv', index=False)