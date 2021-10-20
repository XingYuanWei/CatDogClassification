#!/usr/bin/env python
# -#-coding:utf-8 -*-
# author:魏兴源
# datetime:2021/10/18  11:05:52
# software:PyCharm


# 1.加载库
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 2.定义超参数
BATCH_SIZE = 16  # 每批处理的数据
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 放在cuda或者cpu上训练
EPOCHS = 15  # 训练数据集的轮次
modellr = 1e-3

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

# 损失函数,交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 使用预训练模型
resnet_model = torchvision.models.resnet50(pretrained=True)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 2)
resnet_model.to(DEVICE)
# 选择简单暴力的Adam优化器，学习率调低
optimizer = optim.Adam(resnet_model.parameters(), lr=modellr)


# 更新学习率的方法
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 50))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew


train_loss_list = []
train_accuracy_list = []
test_loss_list = []
test_accuracy_list = []
train_iteration_list = []
test_iteration_list = []


# 定义训练方法
def train(model, device, train_loader, optimizer, epoch):
    iteration = 0
    train_correct = 0.0
    model.train()
    sum_loss = 0.0
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss += print_loss
        train_predict = torch.max(output.data, 1)[1]
        if torch.cuda.is_available():
            train_correct += (train_predict.cuda() == target.cuda()).sum()
        else:
            train_correct += (train_predict == target).sum()
        accuracy = (train_correct / total_num) * 100
        print("Epoch: %d , Batch: %3d , Loss : %.8f,train_correct:%d , train_total:%d , accuracy:%.6f" % (
            epoch + 1, batch_idx + 1, loss.item(), train_correct, total_num, accuracy))
        # 存在集合画图
        if (epoch + 1) == EPOCHS:  # 只画出最后一个epoch时候的准确度变化曲线
            iteration += 1
            train_loss_list.append(loss.item())
            train_iteration_list.append(iteration)
            train_accuracy_list.append(accuracy)


# 定义验证方法
def val(model, device, test_loader, epoch):
    print("=====================预测开始=================================")
    iteration = 0
    model.eval()
    test_loss = 0.0
    correct = 0.0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            if torch.cuda.is_available():
                correct += torch.sum(pred.cuda() == target.cuda())
            else:
                correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
        acc = correct / total_num * 100
        avg_loss = test_loss / len(test_loader)
        """
            因为调用这个方法的时候就是每次结束训练一次之后调用
        """
        # iteration += 1
        # 存入集合准备画图
        test_loss_list.append(avg_loss)
        test_accuracy_list.append(acc)
        test_iteration_list.append(epoch)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
            avg_loss, correct, len(test_loader.dataset), acc))


# 训练
for epoch in range(EPOCHS):
    train(resnet_model, DEVICE, train_loader, optimizer, epoch)
    val(resnet_model, DEVICE, test_loader, epoch)
    # torch.save(resnet_model, 'model.pth')  # 保存模型

# 可视化测试机的loss和accuracy
plt.figure(1)
plt.plot(test_iteration_list, test_loss_list)
plt.title("ResNet50 test loss")
plt.ylabel("loss")
plt.xlabel("Number of test iteration")
plt.show()

plt.figure(2)
plt.plot(test_iteration_list, test_accuracy_list)
plt.title("ResNet50 test accuracy")
plt.xlabel("Number of test iteration")
plt.ylabel("accuracy")
plt.show()

# 可视化训练集loss和accuracy
plt.figure(3)
plt.plot(train_iteration_list, train_loss_list)
plt.title("ResNet50 train loss")
plt.xlabel("Number of train iteration")
plt.ylabel("accuracy")
plt.show()

plt.figure(4)
plt.plot(train_iteration_list, train_accuracy_list)
plt.title("ResNet50 train accuracy")
plt.xlabel("Number of train iteration")
plt.ylabel("accuracy")
plt.show()
