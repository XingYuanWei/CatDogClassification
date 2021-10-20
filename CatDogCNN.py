#!/usr/bin/env python
# -#-coding:utf-8 -*-
# author:魏兴源
# datetime:2021/10/14  8:52:19
# software:PyCharm

"""
模型1：Pytorch CNN 实现流程
    1.图片数据处理，加载数据集
    2.使得数据集可迭代（每次读取一个Batch）
    3.创建模型类
    4.初始化模型类
    5.初始化损失类
    6.训练模型
"""

# 1.加载库
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 2.定义超参数
BATCH_SIZE = 16  # 每批处理的数据
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 放在cuda或者cpu上训练
EPOCHS = 15  # 训练数据集的轮次
LEARNING_RATE = 1e-3

# 3.构建pipeline，对图像做处理
pipeline = transforms.Compose([
    # 分辨率重置为256
    transforms.Resize(256),
    # 对加载的图像作归一化处理， 并裁剪为[224x224x3]大小的图像(因为这图片像素不一致直接统一)
    transforms.CenterCrop(224),
    # 将图片转成tensor
    transforms.ToTensor(),
    # 正则化，模型出现过拟合现象时，降低模型复杂度
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 图片路径(训练图片和测试图片的)
base_dir_train = 'data/train'
base_dir_test = 'data/val'
# 打印一下训练图片猫狗各多少张图片
print('train dogs total images : %d' % (len(os.listdir(base_dir_train + '\\dog'))))
print('train cats total images : %d' % (len(os.listdir(base_dir_train + '\\cat'))))
print('test cats total images : %d' % (len(os.listdir(base_dir_test + '\\cat'))))
print('test dogs total images : %d' % (len(os.listdir(base_dir_test + '\\dog'))))

# 4. 加载数据集
"""
     训练集,猫是0,狗是1，ImageFolder方法自己分类的，关于ImageFolder详见: 
    https://blog.csdn.net/weixin_42147780/article/details/102683053?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-2.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-2.no_search_link
"""
train_dataset = datasets.ImageFolder(root=base_dir_train, transform=pipeline)
print("train_dataset=" + repr(train_dataset[1][0].size()))
print("train_dataset.class_to_idx=" + repr(train_dataset.class_to_idx))
# 创建训练集的可迭代对象，一个batch_size地读取数据,shuffle设为True表示随机打乱顺序读取
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 测试集
test_dataset = datasets.ImageFolder(root=base_dir_test, transform=pipeline)
print(test_dataset)
print("test_dataset=" + repr(test_dataset[1][0].size()))
print("test_dataset.class_to_idx=" + repr(test_dataset.class_to_idx))
# 创建测试集的可迭代对象，一个batch_size地读取数据
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 获得一批测试集的数据
images, labels = next(iter(test_loader))
print(images.shape)
print(labels.shape)


# 5.定义函数，显示一批图片
def imShow(inp, title=None):
    # tensor转成numpy,tranpose转成(通道数,长,宽)
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])  # 均值
    std = np.array([0.229, 0.224, 0.225])  # 标准差
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)  # 像素值限制在0-1之间
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# 网格显示
out = torchvision.utils.make_grid(images)
imShow(out)


class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # 卷积层1：输入是224*224*3 计算(224-5)/1+1=220 即通过Conv1输出的结果是220
        self.conv1 = nn.Conv2d(3, 6, 5)  # input:3 output6 kernel:5
        # 池化层：输入是220*220*6 窗口2*2  计算(220-0)/2=110 那么通过max_pooling层输出的是110*110*6
        self.pool = nn.MaxPool2d(2, 2)
        # 卷积层2， 输入是220*220*6，计算（110 - 5）/ 1 + 1 = 106，那么通过conv2输出的结果是106*106*16
        self.conv2 = nn.Conv2d(6, 16, 5)  # input:6, output:16, kernel:5
        # 全连接层1
        self.fc1 = nn.Linear(16 * 53 * 53, 1024)  # input:16*53*53, output:1024
        # 全连接层2
        self.fc2 = nn.Linear(1024, 512)  # input:1024, output:512
        # 全连接层3
        self.fc3 = nn.Linear(512, 2)  # input:512, output:2
        # dropout 层
        self.dropout = nn.Dropout(p=0.2)  # 因为数据较少，丢掉20%的神经元，防止过拟合

    def forward(self, x):
        # 卷积1
        """
        224x224x3 --> 110x110x6 -->106x106*6
        """
        x = self.pool(F.relu(self.conv1(x)))
        # 卷积2
        """
        106x106x6 --> 53x53x16 
        """
        x = self.pool(F.relu(self.conv2(x)))
        # 改变shape
        x = x.view(-1, 16 * 53 * 53)
        # 全连接层1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # 全连接层2
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # 全连接层3                                 
        x = F.relu(self.fc3(x))
        # print("x size", x.shape)  # x size torch.Size([16, 2])
        return x


# 创建模型，并部署到device中
cnn_model = CNN_Model().to(DEVICE)
if os.path.exists('../CatDogClassifier/Model/model.pt'):
    net = torch.load('../CatDogClassifier/Model/model.pt')

# 优化器
optimizer = optim.SGD(cnn_model.parameters(), lr=LEARNING_RATE, momentum=0.9)
# optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3, betas=(0.9, 0.99))
# 损失函数,交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 把损失，准确度，迭代都记录出list，然后讲loss和准确度画出图像
train_loss_list = []
train_accuracy_list = []
train_iteration_list = []
test_loss_list = []
test_accuracy_list = []
test = test_iteration_list = []

iteration = 0
# for i, (imgs, labels) in enumerate(test_loader):
#     # print("imgs=" + repr(imgs))
#     print("labels=" + repr(labels))
#     print("i=" + repr(i))


# 6.训练
for epoch in range(EPOCHS):
    # 用来显示训练的loss correct等
    train_correct = 0.0
    train_total = 0.0
    for i, (imgs, labels) in enumerate(train_loader):
        # 声明训练，loss等只能在train mode下进行运算
        cnn_model.train()
        # 把训练的数据集合都扔到对应的设备去
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        # 防止梯度爆炸，梯度清零
        optimizer.zero_grad()
        # 前向传播
        cnn_model = cnn_model.cuda()  # 这里要从cuda()中取得，不然前面都放在cuda后面放在cpu，会报错，报“不在同一个设备的错误"
        output = cnn_model(imgs)
        # 计算损失
        loss = criterion(output, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 计算训练时候的准确度
        train_predict = torch.max(output.data, 1)[1]
        if torch.cuda.is_available():
            train_correct += (train_predict.cuda() == labels.cuda()).sum()
        else:
            train_correct += (train_predict == labels).sum()
        train_total += labels.size(0)
        accuracy = train_correct / train_total * 100.0
        # 只画出最后一次epoch的
        if (epoch + 1) == EPOCHS:
            # 迭代计数器++
            iteration += 1
            train_accuracy_list.append(accuracy)
            train_iteration_list.append(iteration)
            train_loss_list.append(loss)
        # 打印信息
        print("Training---->Epoch :%d , Batch : %5d , Loss : %.8f,train_correct:%d,train_total:%d,accuracy:%.6f" % (
            epoch + 1, i + 1, loss.item(), train_correct, train_total, accuracy))
        
    # 每次训练完一个epoch后在测试运行一次
    # ========================== 在测试集运行===============================================
    print("==========================预测开始===========================")
    cnn_model.eval()
    # 验证accuracy
    correct = 0.0
    total = 0.0
    # 迭代测试集 获取数据 预测
    for j, (datas, targets) in enumerate(test_loader):
        datas, targets = datas.to(DEVICE), targets.to(DEVICE)
        # 模型预测
        outputs = cnn_model(datas)
        # 获取测试概率最大值的下标
        predicted = torch.max(outputs.data, 1)[1]
        # 统计计算测试集合
        total += targets.size(0)
        if torch.cuda.is_available():
            correct += (predicted.cuda() == targets.cuda()).sum()
            # print("predicted.cuda()=" + repr(predicted.cuda()))
            # print("labels.cuda()=" + repr(targets.cuda()))
        else:
            correct += (predicted == targets).sum()
    accuracy = correct / total * 100.0
    test_accuracy_list.append(accuracy)
    test_loss_list.append(loss.item())
    test_iteration_list.append(epoch)
    print(
        "TEST--->loop : {}, Loss : {}, correct:{}, total:{}, Accuracy : {}".format(iteration+1, loss.item(), correct,
                                                                                       total, accuracy))

# 可视化训练集loss
plt.figure(1)
plt.plot(train_iteration_list, train_loss_list)
plt.xlabel("number of iteration")
plt.ylabel("loss")
plt.title("CNN train loss")
plt.show()

# 可视化训练集accuracy
plt.figure(2)
plt.plot(train_iteration_list, train_accuracy_list)
plt.xlabel('number of iteration')
plt.ylabel('accuracy')
plt.title('CNN train accuracy')
plt.show()

# 可视化测试集accuracy
plt.figure(3)
plt.plot(test_iteration_list, test_accuracy_list)
plt.xlabel('number of iteration')
plt.ylabel('accuracy')
plt.title('CNN test accuracy')
plt.show()

# 可视化测试集loss
plt.figure(4)
plt.plot(test_iteration_list, test_loss_list)
plt.xlabel('number of iteration')
plt.ylabel('accuracy')
plt.title('CNN test loss')
plt.show()

