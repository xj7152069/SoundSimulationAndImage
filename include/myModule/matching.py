#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math

class MaskedModel(nn.Module):
    def __init__(self, nIn, nOut):
        super(MaskedModel, self).__init__()
        self.nfc = nn.Linear(nIn, nOut)

    def forward(self, input, mask):
        return F.linear(input, self.nfc.weight*mask, self.nfc.bias)
    
    
def wienerMultiTraceTransformer\
    (origLocal, mulLocal, maskIn, nw, nHead, numTrain=100, lr=0.01):
    # use CPU or GPU
    myDevice=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(myDevice)
    # 创建网络实例
    model = MaskedModel(nw*nHead, 1*nHead)
    model.to(myDevice)
    # 打印模型结构
    # print(model)
    # 定义损失函数和优化器
    criterion = nn.MSELoss().to(myDevice)  # 均方误差损失函数
    optimizer= optim.Adam(model.parameters(), lr) # Adam 优化器

    X = torch.tensor(mulLocal).to(myDevice)  # Nt 个样本，nw 个特征
    Y = torch.tensor(origLocal).to(myDevice)  # Nt 个目标值
    mask = torch.tensor(maskIn).to(myDevice)

    # 训练循环
    for epoch in range(numTrain):  # 训练 100 轮
        optimizer.zero_grad()  # 清空之前的梯度
        output = model(X,mask)  # 前向传播
        loss = criterion(output, Y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 每 10 轮输出一次损失
        if (epoch+1) % (1) == 0:
            print(f'Epoch [{epoch+1}/{numTrain}], Loss: {loss.item():.4f}')
    
    #Output Result:
    Z=model(X,mask)
    Z=Z.cpu()
    matchLocal=Z.detach().numpy()
    model.nfc.reset_parameters()
    return matchLocal

#################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary

class unetConv2(nn.Module):
    def __init__(self,in_size,out_size,is_batchnorm):
        super(unetConv2,self).__init__()

        if is_batchnorm:
            self.conv1=nn.Sequential(
                nn.Conv2d(in_size,out_size,kernel_size=3,stride=1,padding=0),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
            )
            self.conv2=nn.Sequential(
                nn.Conv2d(out_size,out_size,kernel_size=3,stride=1,padding=0),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1=nn.Sequential(
                nn.Conv2d(in_size,out_size,kernel_size=3,stride=1,padding=0),
                nn.ReLU(inplace=True),
            )
            self.conv2=nn.Sequential(
                nn.Conv2d(out_size,out_size,kernel_size=3,stride=1,padding=0),
                nn.ReLU(inplace=True)
            )
    def forward(self, inputs):
        outputs=self.conv1(inputs)
        outputs=self.conv2(outputs)

        return outputs

class unetUp(nn.Module):
    def __init__(self,in_size,out_size,is_deconv):
        super(unetUp,self).__init__()
        self.conv=unetConv2(in_size,out_size,False)
        if is_deconv:
            self.up=nn.ConvTranspose2d(in_size,out_size,kernel_size=2,stride=2)
        else:
            self.up=nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1,inputs2):
        outputs2=self.up(inputs2)
        offset=outputs2.size()[2]-inputs1.size()[2]
        padding=2*[offset//2,offset//2]
        outputs1=F.pad(inputs1,padding)     #padding is negative, size become smaller

        return self.conv(torch.cat([outputs1,outputs2],1))

class unet(nn.Module):
    def __init__(self,feature_scale=4,n_classes=21,is_deconv=True,in_channels=3,is_batchnorm=True):
        super(unet,self).__init__()
        self.is_deconv=is_deconv
        self.in_channels=in_channels
        self.is_batchnorm=is_batchnorm
        self.feature_scale=feature_scale

        filters=[64,128,256,512,1024]
        filters=[int(x/self.feature_scale) for x in filters]

        #downsample
        self.conv1=unetConv2(self.in_channels,filters[0],self.is_batchnorm)
        self.maxpool1=nn.MaxPool2d(kernel_size=2)

        self.conv2=unetConv2(filters[0],filters[1],self.is_batchnorm)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)

        self.conv3=unetConv2(filters[1],filters[2],self.is_batchnorm)
        self.maxpool3=nn.MaxPool2d(kernel_size=2)

        self.conv4=unetConv2(filters[2],filters[3],self.is_batchnorm)
        self.maxpool4=nn.MaxPool2d(kernel_size=2)

        self.center=unetConv2(filters[3],filters[4],self.is_batchnorm)

        #umsampling
        self.up_concat4=unetUp(filters[4],filters[3],self.is_deconv)
        self.up_concat3=unetUp(filters[3],filters[2],self.is_deconv)
        self.up_concat2=unetUp(filters[2],filters[1],self.is_deconv)
        self.up_concat1=unetUp(filters[1],filters[0],self.is_deconv)

        #final conv (without and concat)
        self.final=nn.Conv2d(filters[0],n_classes,kernel_size=1)

    def forward(self, inputs):
        conv1=self.conv1(inputs)
        maxpool1=self.maxpool1(conv1)

        conv2=self.conv2(maxpool1)
        maxpool2=self.maxpool2(conv2)

        conv3=self.conv3(maxpool2)
        maxpool3=self.maxpool3(conv3)

        conv4=self.conv4(maxpool3)
        maxpool4=self.maxpool4(conv4)

        center=self.center(maxpool4)
        up4=self.up_concat4(conv4,center)
        up3=self.up_concat3(conv3,up4)
        up2=self.up_concat2(conv2,up3)
        up1=self.up_concat1(conv1,up2)

        final=self.final(up1)

        return final

if __name__=="__main__":
    model=unet(feature_scale=1)
    #print(summary(model,(3,572,572)))
    print((model,(3,572,572)))
#################################################################################

def matchingOneBlock(orig2d, mul2d, \
    halfw=2, halftrace=10, halfntl=30, gapntl=10, numTrain=100, lr=0.01):

    shape=orig2d.shape
    nt=shape[0]
    nx=shape[1]

    origMax=orig2d.max()
    orig2d=orig2d/orig2d.max()
    mul2d=mul2d/mul2d.max()

    dem2d=np.zeros([nt,nx],np.float32)
    match2d=np.zeros([nt,nx],np.float32)
    origRe2d=np.zeros([nt,nx],np.float32)
    mulRe2d=np.zeros([nt,nx],np.float32)

    nw=2*halfw+1
    ntrace=2*halftrace+1
    ntl=2*halfntl+1

    dem2d.fill(0.001)
    match2d.fill(0.0)
    origRe2d.fill(0.0)
    mulRe2d.fill(0.0)

    itl=halfw
    ix=halftrace
    nHead=(nx-halftrace-ix+1)*(math.floor((nt-ntl-halfw-itl)/gapntl)+1)
    mulLocal=np.zeros([ntl*ntrace,nw*nHead],np.float32)
    origLocal=np.zeros([ntl*ntrace,1*nHead],np.float32)
    maskIn=np.zeros([1*nHead,nw*nHead],np.float32)

    iHead=0
    while ix<nx-halftrace:
        itl=halfw
        while itl<nt-ntl-halfw:
            for itrace in range(ntrace):
                origLocal[itrace*ntl:(itrace+1)*ntl,iHead:iHead+1]\
                    =np.copy(orig2d[itl:itl+ntl,ix-halftrace+itrace:ix-halftrace+itrace+1])
                for iw in range(nw):
                    mulLocal[itrace*ntl:(itrace+1)*ntl,iHead*nw+iw:iHead*nw+iw+1]\
                        =np.copy(mul2d[itl+iw-halfw:itl+ntl+iw-halfw,\
                            ix-halftrace+itrace:ix-halftrace+itrace+1])
            maskIn[iHead:iHead+1,iHead*nw:(iHead+1)*(nw)]\
                =np.copy(maskIn[iHead:iHead+1,iHead*nw:(iHead+1)*(nw)]+1)
            iHead=iHead+1
            itl=itl+gapntl
        ix=ix+1
        
    print(mulLocal.shape)
    print(maskIn.shape)
    print(origLocal.shape)
    # Wiener Matching:
    matchLocal=wienerMultiTraceTransformer\
        (origLocal, mulLocal, maskIn, nw, nHead, numTrain, lr)

    #print(matchLocal.shape)

    itl=halfw
    iHead=0
    ix=halftrace

    while ix<nx-halftrace:
        itl=halfw
        while itl<nt-ntl-halfw:
            match2d[itl:itl+ntl,ix:ix+1]=np.copy(match2d[itl:itl+ntl,ix:ix+1]\
                +matchLocal[halftrace*ntl:(halftrace+1)*ntl,iHead:iHead+1])

            dem2d[itl:itl+ntl,ix:ix+1]=np.copy(dem2d[itl:itl+ntl,ix:ix+1]+1.0)
            mulRe2d[itl:itl+ntl,ix:ix+1]=np.copy(mulRe2d[itl:itl+ntl,ix:ix+1]\
                +mulLocal[halftrace*ntl:(halftrace+1)*ntl,iHead*nw+halfw:iHead*nw+halfw+1])
            origRe2d[itl:itl+ntl,ix:ix+1]=np.copy(origRe2d[itl:itl+ntl,ix:ix+1]\
                +origLocal[halftrace*ntl:(halftrace+1)*ntl,iHead:iHead+1])
            iHead=iHead+1
            itl=itl+gapntl
        ix=ix+1

    mulRe2d=mulRe2d/dem2d
    origRe2d=origRe2d/dem2d
    match2d=match2d/dem2d
    dem2d=np.copy(orig2d)
    dem2d=dem2d-match2d
    dem2d=dem2d*origMax
    match2d=match2d*origMax

    return dem2d, match2d









