import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,dilation=1):
        super(unit_tcn, self).__init__()
        new_ks = (kernel_size-1)*dilation+1
        pad = int((new_ks - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1),dilation=(dilation,1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1)) #在内部压缩了通道数
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
               nn.Conv2d(in_channels, out_channels, 1),
               nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)

class part_pooling(nn.Module):
    def __init__(self,inchannels,partion_metrix):
        super(part_pooling,self).__init__()
        self.partion_metrix = Variable(torch.from_numpy(np.repeat(np.expand_dims(partion_metrix, 0), inchannels, axis=0)
                                                        .astype(np.float32)), requires_grad=False)
        self.joint_weight = nn.Parameter(torch.from_numpy(
            np.where(self.partion_metrix==0,np.zeros_like(self.partion_metrix),np.ones_like(self.partion_metrix))
            .astype(np.float32)), requires_grad=True)
        self.bn = nn.BatchNorm2d(inchannels)
        bn_init(self.bn, 1)
    def forward(self, x):
        n,c,t,v = x.size()
        M = self.partion_metrix.cuda(x.get_device()) * self.joint_weight
        M = M.unsqueeze(0)
        x = torch.matmul(x,M)
        x= x.view(n,c, t,-1)
        x = self.bn(x)
        return  x






class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1,kt=9,dilation=1,deform=False,partion_m =None ,residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        if deform:
            self.tcn1 = DeformTConv(out_channels,out_channels,stride=stride)
        else:
            self.tcn1 = unit_tcn(out_channels, out_channels, kernel_size=kt,stride=stride,dilation=dilation)
        self.relu = nn.ReLU()
        self.partion = part_pooling(out_channels,partion_m) if partion_m is not None else None
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        if self.partion is not None:
            x = self.tcn1(self.gcn1(x)) + self.residual(x)
            x = self.relu(x)
            x = self.partion(x)
        else:
            x = self.tcn1(self.gcn1(x)) + self.residual(x)
            x = self.relu(x)
        return x


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,dropout=0):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        A = self.graph.A
        M = self.graph.M
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.l1 = TCN_GCN_unit(in_channels, 64, A[0],kt=1)
        self.l2 = TCN_GCN_unit(64, 64,A[0],kt=3)
        self.l3 = TCN_GCN_unit(64, 128, A[0], stride=2,partion_m=M[0])
        self.l4 = TCN_GCN_unit(128, 128, A[1])
        self.l5 = TCN_GCN_unit(128, 256, A[1], stride=2,partion_m=M[1])
        self.l6 = TCN_GCN_unit(256, 256, A[2],dilation=5)
        self.drop_out = nn.Dropout(dropout)
        self.fc = nn.Linear(448,num_class)

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / 448))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        c_new = x.size(1)
        y1 = x.view(N, M, c_new, -1)
        y1 = y1.mean(3).mean(1)
        x = self.l3(x)
        x = self.l4(x)
        c_new = x.size(1)
        y2 = x.view(N, M, c_new, -1)
        y2 = y2.mean(3).mean(1)
        x = self.l5(x)
        x = self.l6(x)
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        x = torch.cat((x,y2,y1),dim=1)
       # x = torch.cat((x, y2), dim=1)
        x =self.drop_out(x)
        x = self.fc(x)
        return x
