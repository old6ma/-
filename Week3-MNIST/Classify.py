import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

import CNN
import paras

# 数据增强操作，把图像数据转化为张量
train_transform = torchvision.transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(5),
                                                  transforms.Normalize((0.1307,), (0.3081,))])
test_transform = torchvision.transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# 下载MNIST数据集，并用数据增强进行处理
trainData = torchvision.datasets.MNIST('./data/', download=True, transform=train_transform)
testData = torchvision.datasets.MNIST('./data/', train=False, download=True, transform=test_transform)

# 准备数据集
train_loader = torch.utils.data.DataLoader(dataset=trainData, batch_size=paras.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testData, batch_size=paras.batch_size)

# 初始化网络实例
mymodel = CNN.CNN2()
# GPU加速
mymodel = mymodel.cuda()
# 定义交叉熵损失函数
loss = nn.CrossEntropyLoss()
learn_r = 0.005
# 使用Adam优化器
optimizer = torch.optim.Adam(mymodel.parameters(), lr=learn_r)

# 通过checkpoint保存训练好的模型，下次可以直接使用
use_check_point = ''
# 加载checkpoint
if os.path.getsize("checkpoint.pth") == 0:
    print("checkppoint is empty")
else:
    UseCheckpoint = input("Use checkpoint? please input yes or no:")
    use_check_point = UseCheckpoint
    if UseCheckpoint == "yes":
        print("loading checkpoint......")
        checkpoint = torch.load("checkpoint.pth")
        mymodel.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        Train_acc = checkpoint['Train Acc']
        Train_loss = checkpoint['Train Loss']
        print("Successfully loaded checkpoint......")
        # 输出当前保存的模型的性能
        print('The model has been trained for %d epochs Train Acc: %3.6f Loss: %3.6f ' % (
            checkpoint['epoch'] + 1, Train_acc, Train_loss))
        # 选择继续训练的epoch数，输入0可以跳过训练，直接进行测试
        new_epoch = int(input("How many epochs do you want?"))
        paras.epoch = new_epoch

for epoch in range(paras.epoch):
    if epoch % 40 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print("This is %d epoch,lr is %f." % (epoch, current_lr))
    # 定义变量，包括计时器，训练准确率，loss值
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    mymodel.train()
    for i, data in enumerate(train_loader):  # data[0]是训练集，data[1]则是标签
        # 优化器参数先清零
        optimizer.zero_grad()
        # 得到训练的预测结果
        train_pred = mymodel(data[0].cuda())
        # 计算loss
        batch_loss = loss(train_pred, data[1].cuda())
        # 反向传播更新参数
        batch_loss.backward()
        optimizer.step()
        # 计算当前准确率以及loss
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy()) / data[1].shape[0]
        train_loss += batch_loss.item()
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f ' %
          (epoch + 1, paras.epoch, time.time() - epoch_start_time,
           train_acc / paras.batch_size, train_loss / (paras.batch_size * len(train_loader))))
    # 保存当前模型状态
    if use_check_point == "yes":
        checkpoint = torch.load("checkpoint.pth")
        true_epoch = checkpoint['epoch'] + 1 + epoch
        torch.save({
            'epoch': true_epoch,
            'Train Acc': train_acc / (paras.batch_size * len(train_loader)),
            'Train Loss': train_loss / (paras.batch_size * len(train_loader)),
            'model_state': mymodel.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, 'checkpoint.pth')
    else:
        torch.save({
            'epoch': epoch,
            'Train Acc': train_acc / (paras.batch_size * len(train_loader)),
            'Train Loss': train_loss / (paras.batch_size * len(train_loader)),
            'model_state': mymodel.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, 'checkpoint.pth')

# 测试
mymodel.eval()
prediction = []
with torch.no_grad():
    test_acc = 0.0
    test_loss = 0.0
    for i, data in enumerate(test_loader):
        test_pred = mymodel(data[0].cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        test_acc += np.sum(np.argmax(test_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
    print("final accuracy is %f" % (test_acc / 10000))
