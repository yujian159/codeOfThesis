import cv2
import numpy as np
import glob

# 预处理 构建数据矩阵
images = glob.glob(r'.\orl_MBLBP3\*.jpg')
X = []
for img in images:
    img = cv2.imread(img, 0)
    temp = np.resize(img, (img.shape[0] * img.shape[1], 1))
    X.append(temp.T)
X = np.array(X).squeeze().T
print(X.shape, X.shape[1])  # (10304, 400) 400

# 10轮
correct_sum = 0
for epoch in range(10):
    train_data = X[:, [x for x in list(range(X.shape[1])) if x not in list(range(epoch, X.shape[1], 10))]]
    test_data = X[:, list(range(epoch, X.shape[1], 10))]

    # train
    u = np.sum(train_data, axis=1) / train_data.shape[1]  # 求均值向量

    u = u[:, np.newaxis]
    C = train_data - u  # 中心化后数据矩阵
    Covariance = np.dot(C.T, C)  # 构建协方差矩阵，一般为C .* C.T，但是构造这种类型可减少运算量
    eigvalue, eigvector = np.linalg.eig(Covariance)  # 由协方差矩阵求解特征值、特征向量
    real_eigvector = np.dot(C, eigvector)  # 通过之前的构造来恢复真正协方差矩阵对应的特征向量
    sort = np.argsort(-eigvalue)  # 将特征值从大到小怕排序，得到排序后对于原索引
    P = real_eigvector.T[sort[0:100]]  # 对于排序构造特征向量，取前面较大权重值
    Y = []
    for i in range(train_data.shape[1]):
        temp = train_data[:, i, np.newaxis]
        Y.append(np.dot(P, temp - u))  # 构建每幅图像投影后的值，构造查找表

    # test
    correct = 0
    for index in range(test_data.shape[1]):
        img_test = test_data[:, index, np.newaxis]  # 从测试集提取单张人脸
        Result = np.dot(P, img_test - u)  # 计算待识别的人脸的投影
        a = np.sum(abs(Y - Result), axis=1).argmin()  # 遍历搜索匹配
        if index * 9 <= a < (index + 1) * 9:  # 若索引在宽度为9的区间内则为该人脸，视为匹配正确
            correct += 1
    print('Epoch{} correct rate: {}%'.format(epoch, correct / 40 * 100))
    correct_sum += correct

print('Final  correct rate: {}%'.format(correct_sum / 4))






