#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class SimpleLiner(nn.Module):
    def __init__(self, nIn, nOut):
        super(SimpleLiner, self).__init__()
        self.fc = nn.Linear(nIn, nOut)
   
    def forward(self, x):
        x = self.fc(x)
        return x

# 定义一个简单的全连接神经网络
class SimpleNN(nn.Module):
    def __init__(self, nIn, nOut):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(nIn, nIn)  # 输入层到隐藏层
        self.fc2 = nn.Linear(nIn, nOut)  # 隐藏层到输出层
   
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU 激活函数
        x = self.fc2(x)
        return x

def wienerMultiTrace(origLocal, mulLocal, nw,\
                     numTrain=20, lr=0.001):
    # 创建网络实例
    model = SimpleLiner(nw,1)
    # model = SimpleNN(nw,1)
    # 打印模型结构
    # print(model)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr)  # Adam 优化器

    X = torch.tensor(mulLocal)  # 1000 个样本，3 个特征
    Y = torch.tensor(origLocal)  # 1000 个目标值
    I=torch.eye(nw, nw)*0.1
    '''
    print(X.shape)
    print(model.fc.weight.shape)
    print(Y.shape)
    XT=torch.transpose(X,0,1)
    XTX=torch.matmul(XT,X)
    XTX=XTX+I*XTX.max()*10.0
    XTY=torch.matmul(XT,Y)
    '''
    # 训练循环
    for epoch in range(numTrain):  # 训练 100 轮
        optimizer.zero_grad()  # 清空之前的梯度
        output = model(X)  # 前向传播
        loss = criterion(output, Y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 每 10 轮输出一次损失
        if (epoch+1) % (numTrain) == 0:
            print(f'Epoch [{epoch+1}/{numTrain}], Loss: {loss.item():.4f}')
    
    #Output Result:
    Z=model(X)
    matchLocal=Z.detach().numpy()
    model.fc.reset_parameters()
    #model.fc1.reset_parameters()
    #model.fc2.reset_parameters()
    return matchLocal


if __name__ == "__main__":
    # 创建网络实例
    model = SimpleNN()

    # 打印模型结构
    print(model)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

    # 假设我们有训练数据 X 和 Y
    X = torch.randn(10, 3)  # 10 个样本，2 个特征
    Y = torch.randn(10, 1)  # 10 个目标值

    # 5. 训练循环
    for epoch in range(100):  # 训练 100 轮
        optimizer.zero_grad()  # 清空之前的梯度
        output = model(X)  # 前向传播
        loss = criterion(output, Y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
    
        # 每 10 轮输出一次损失
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')




