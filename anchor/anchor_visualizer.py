import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

DATA_PATH = "./data_03_10/labels"
total_wh = []
xy_list = []
for file_name in tqdm(os.listdir(DATA_PATH)):
    if not file_name.endswith(".txt"):
        continue
    with open(os.path.join(DATA_PATH, file_name), "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()  # 删除字符串前后的特殊字符
            label = line.split()  # 默认按空格分割
            if len(label) != 5:
                continue
            total_wh.append([float(label[3]), float(label[4])])
            xy_list.append([float(label[1]), float(label[2])])

total_wh = np.array(total_wh)
xy_list = np.array(xy_list)
print(xy_list.min(), xy_list.max())

k_means = KMeans(n_clusters=4)
k_means.fit(total_wh)

plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], s=100, c='red')
plt.scatter(total_wh[:, 0],total_wh[:, 1], s=0.1)
# plt.scatter(xy_list[:, 0],xy_list[:, 1], s=0.3, c='green')
plt.show()

cluster_centers: np.ndarray = k_means.cluster_centers_
# cluster_centers = cluster_centers.tolist()
# breakpoint()
cluster_centers = sorted(cluster_centers, key=lambda x: x[0] * x[1])
print(np.array(cluster_centers))
'''
[[0.06196073 0.08778166]
 [0.1406642  0.41936443]
 [0.15771551 0.27112423]
 [0.19960276 0.33541366]
 [0.12430502 0.21680116]
 [0.10700961 0.15608912]
 [0.25727868 0.43221634]
 [0.25117803 0.27420782]
 [0.28020981 0.66155584]]

 [[0.07711985 0.10994485]
 [0.12846589 0.21537308]
 [0.19778088 0.31619958]
 [0.23630117 0.48031218]]
'''
