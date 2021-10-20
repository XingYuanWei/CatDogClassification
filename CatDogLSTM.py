#!/usr/bin/env python
# -#-coding:utf-8 -*-
# author:魏兴源
# datetime:2021/10/17  21:35:59
# software:PyCharm


"""
模型1：Pytorch LSTM 实现流程
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
BATCH_SIZE = 32  # 每批处理的数据
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 放在cuda或者cpu上训练
EPOCHS = 15  # 训练数据集的轮次

# 3.构建pipeline，对图像做处理
pipeline = transforms.Compose([
    # 彩色图像转灰度图像num_output_channels默认1
    # transforms.Grayscale(num_output_channels=1),
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
# print(test_dataset)
print("test_dataset=" + repr(test_dataset[1][0].size()))
print("test_dataset.class_to_idx=" + repr(test_dataset.class_to_idx))
# 创建测试集的可迭代对象，一个batch_size地读取数据
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 获得一批测试集的数据
images, labels = next(iter(test_loader))
print("images shape", images.shape)
print("labels shape", labels.shape)


# 5.定义函数，显示一批图片
def imShow(inp, title=None):
    # tensor转成numpy,transpose转成(通道数,长,宽)
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


# 6.定义LSTM网络
class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM_Model, self).__init__()  # 初始化父类构造方法
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # 构建LSTM模型
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化隐藏层状态全为0
        # (layer_dim,batch_size,hidden_dim)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)
        # 初始化cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)
        x = x.view(x.size(0), 1, -1)
        # 分离隐藏状态 避免梯度爆炸
        '''
            RNN只有一个状态，而LSTM有两个状态，所以两个状态都要分离
        '''
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 只需要最后一层隐层的状态
        out = self.fc(out[:, -1, :])
        return out


# 7.初始化模型
input_dim = 150528  # 输入维度(输入的节点数量)
hidden_dim = 50  # 隐藏层的维度(每个隐藏层的节点数)
layer_dim = 2  # 2层LSTM(隐藏层的数量 2层)
out_dim = 2  # 输出维度
rnn_model = LSTM_Model(input_dim, hidden_dim, layer_dim, out_dim)

# 8.输出模型参数信息
length = len(list(rnn_model.parameters()))
print(length)

# 9.输出模型参数信息
length = len(list(rnn_model.parameters()))
print(length)

# 优化器
# optimizer = optim.SGD(rnn_model.parameters(), lr=1e-3, momentum=0.9)
optimizer = optim.Adam(rnn_model.parameters(), lr=1e-3, betas=(0.9, 0.99))

# 损失函数,交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 把损失，准确度，迭代都记录出list，然后讲loss和准确度画出图像
sequence_dim = 53
train_loss_list = []
train_accuracy_list = []
train_iteration_list = []

test_loss_list = []
test_accuracy_list = []
test_iteration_list = []

iteration = 0
# for i, (imgs, labels) in enumerate(test_loader):
#     # print("imgs=" + repr(imgs))
#     print("labels=" + repr(labels))
#     print("i=" + repr(i))


# 训练
# """
for epoch in range(EPOCHS):
    # 用来显示训练的loss correct等
    train_correct = 0.0
    train_total = 0.0
    for i, (imgs, labels) in enumerate(train_loader):
        # 声明训练，loss等只能在train mode下进行运算
        rnn_model.train()
        # 把训练的数据集合都扔到对应的设备去
        # imgs = imgs.view(-1,1,sequence_dim, input_dim).requires_grad_().to(DEVICE)
        # print("imgs shape", imgs.shape)
        # print("imgs = ", imgs.data)
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        # 防止梯度爆炸，梯度清零
        optimizer.zero_grad()
        # 前向传播
        rnn_model = rnn_model.cuda()  # 这里要从cuda()中取得，不然前面都放在cuda后面放在cpu，会报错，报“不在同一个设备的错误" Input and parameter tensors are not at the same device, found input tensor at cuda:0 and parameter tensor at cpu
        output = rnn_model(imgs)
        # print("RNN output shape", out.shape)
        # print("label shape", labels.shape)
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
        print("Epoch :%d , Batch : %5d , Loss : %.8f,train_correct:%d,train_total:%d,accuracy:%.6f" % (
            epoch + 1, i + 1, loss.item(), train_correct, train_total, accuracy))
    print("==========================预测开始===========================")
    rnn_model.eval()
    # 验证accuracy
    correct = 0.0
    total = 0.0
    # 迭代测试集 获取数据 预测
    for j, (datas, targets) in enumerate(test_loader):
        datas = datas.to(DEVICE)
        targets = targets.to(DEVICE)
        # datas = datas.view(-1, sequence_dim, input_dim).requires_grad_().to(DEVICE)
        # datas = datas.reshape(datas.size(0), 1, -1)
        # 模型预测
        outputs = rnn_model(datas)
        # 防止梯度爆炸，梯度清零
        optimizer.zero_grad()
        # 获取测试概率最大值的下标
        predicted = torch.max(outputs.data, 1)[1]
        # 统计计算测试集合
        total += targets.size(0)
        if torch.cuda.is_available():
            # print(predicted.cuda() == targets.cuda())
            correct += (predicted.cuda() == targets.cuda()).sum()
            # print("predicted.cuda()=" + repr(predicted.cuda()))
            # print("labels.cuda()=" + repr(targets.cuda()))
        else:
            correct += (predicted == targets).sum()
    accuracy = correct / total * 100.0
    test_accuracy_list.append(accuracy)
    test_loss_list.append(loss.item())
    test_iteration_list.append(iteration)
    print("TEST--->loop : {}, Loss : {}, correct:{}, total:{}, Accuracy : {}".format(iteration, loss.item(),
                                                                                             correct,
                                                                                             total, accuracy))
# 可视化训练集loss
plt.figure(1)
plt.plot(train_iteration_list, train_loss_list)
plt.xlabel("number of iteration")
plt.ylabel("loss")
plt.title("RNN train loss")
plt.show()

# 可视化训练集accuracy
plt.figure(2)
plt.plot(train_iteration_list, train_accuracy_list)
plt.xlabel('number of iteration')
plt.ylabel('accuracy')
plt.title('LSTM train accuracy')
plt.show()

# 可视化测试集loss
plt.figure(3)
plt.plot(test_iteration_list, test_loss_list)
plt.xlabel('number of iteration')
plt.ylabel('loss')
plt.title('LSTM test loss')
plt.show()

# 可视化测试集accuracy
plt.figure(4)
plt.plot(test_iteration_list, test_accuracy_list)
plt.xlabel('number of iteration')
plt.ylabel('accuracy')
plt.title('LSTM test accuracy')
plt.show()
