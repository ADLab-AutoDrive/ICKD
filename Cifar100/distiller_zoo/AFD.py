# Attention-based Feature-level Distillation 
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class nn_bn_relu(nn.Module):
    def __init__(self, nin, nout):
        super(nn_bn_relu, self).__init__()
        self.linear = nn.Linear(nin, nout)
        self.bn = nn.BatchNorm1d(nout)
        self.relu = nn.ReLU(False)

    def forward(self, x, relu=True):
        if relu:
            return self.relu(self.bn(self.linear(x)))
        return self.bn(self.linear(x))

class AFD(nn.Module):
    def __init__(self, args):
        super(AFD, self).__init__()
        self.guide_layers = args.guide_layers
        self.hint_layers = args.hint_layers
        self.attention = Attention(args)

    def forward(self, g_s, g_t):
        g_t = [g_t[i] for i in self.guide_layers]
        g_s = [g_s[i] for i in self.hint_layers]
        loss = self.attention(g_s, g_t)
        return sum(loss)

class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.qk_dim = args.qk_dim
        self.n_t = args.n_t
        self.linear_trans_s = LinearTransformStudent(args)
        self.linear_trans_t = LinearTransformTeacher(args)

        self.p_t = nn.Parameter(torch.Tensor(len(args.t_shapes), args.qk_dim))
        self.p_s = nn.Parameter(torch.Tensor(len(args.s_shapes), args.qk_dim))
        torch.nn.init.xavier_normal_(self.p_t)
        torch.nn.init.xavier_normal_(self.p_s)

    def forward(self, g_s, g_t):
        bilinear_key, h_hat_s_all = self.linear_trans_s(g_s)
        query, h_t_all = self.linear_trans_t(g_t)
        p_logit = torch.matmul(self.p_t, self.p_s.t())
        #import pdb
        #pdb.set_trace()
        #logit = torch.add(torch.einsum('bstq,btq->bts', bilinear_key, query), p_logit) / np.sqrt(self.qk_dim)
        logit = torch.add(torch.bmm(query, bilinear_key.permute(0,2,1)), p_logit) / np.sqrt(self.qk_dim)
        atts = F.softmax(logit, dim=2)  # b x t x s
        loss = []
        for i, (n, h_t) in enumerate(zip(self.n_t, h_t_all)):
            h_hat_s = h_hat_s_all[n]
            diff = self.cal_diff(h_hat_s, h_t, atts[:, i])
            loss.append(diff)
        return loss

    def cal_diff(self, v_s, v_t, att):
        """ 
        #diff = (v_s - v_t.unsqueeze(1)).pow(2)
        v_s = torch.mul(v_s, att.unsqueeze(-1).unsqueeze(-1)).sum(1)
        diff = (v_s - v_t).pow(2)
        b,c,h = diff.size()
        diff = diff.view(b, -1).sum()/(b*c)
        """
        #import pdb
        #pdb.set_trace()
        diff = (v_s - v_t.unsqueeze(1)).pow(2)
        diff = torch.mul(diff, att.unsqueeze(-1).unsqueeze(-1)).sum(1)
        b,c,h = diff.size()
        diff = diff.view(b, -1).sum()/(b*c)
         
        return diff

class LinearTransformTeacher(nn.Module):
    def __init__(self, args):
        super(LinearTransformTeacher, self).__init__()
        self.query_layer = nn.ModuleList([nn_bn_relu(t_shape[1], args.qk_dim) for t_shape in args.t_shapes])

    def forward(self, g_t):
        bs = g_t[0].size(0)
        f_emb = [f_t.view(bs, f_t.size(1), -1) for f_t in g_t]
        f_dis = [torch.bmm(f_d, f_d.permute(0,2,1)) for f_d in f_emb] 
        #channel_mean = [f_t.mean(3).mean(2) for f_t in g_t]
        #import pdb
        #pdb.set_trace()
        #channel_mean = [f_t.view(bs,f_t.size(1),-1).pow(2).mean(-1) for f_t in g_t]
        #spatial_mean = [f_t.pow(2).mean(1).view(bs, -1) for f_t in g_t]
        channel_mean = [F.normalize(f_s, dim=2).mean(-1) for f_s in f_dis]
        query = torch.stack([query_layer(f_t, relu=False) for f_t, query_layer in zip(channel_mean, self.query_layer)],
                            dim=1)
        value = [F.normalize(f_s, dim=2) for f_s in f_dis]
        #import pdb
        #pdb.set_trace()
        return query, value

class LinearTransformStudent(nn.Module):
    def __init__(self, args):
        super(LinearTransformStudent, self).__init__()
        self.t = len(args.t_shapes)
        self.s = len(args.s_shapes)
        self.qk_dim = args.qk_dim
        self.relu = nn.ReLU(inplace=False)
        #self.samplers = nn.ModuleList([Sample(t_shape) for t_shape in args.unique_t_shapes])
        self.embeds = nn.ModuleList([StudentEmbed(args.s_shapes, t_shape) for t_shape in args.unique_t_shapes])
        self.key_layer = nn.ModuleList([nn_bn_relu(s_shape[1], self.qk_dim) for s_shape in args.s_shapes])
        self.bilinear = nn_bn_relu(args.qk_dim, args.qk_dim * len(args.t_shapes))

    def forward(self, g_s):
        bs, ch = g_s[0].size(0), g_s[0].size(1)
        #g_s = [embed(g_s) for embed in self.embeds]
        #g_s = [ for f_s in g_s]
        g_s_emb = [embed(g_s) for embed in self.embeds]
        f_emb_t_s = [[f_ss.view(bs, f_s[0].size(1), -1) for f_ss in f_s] for f_s in g_s_emb]
        f_dis = [[F.normalize(torch.bmm(f_emd, f_emd.permute(0,2,1)), 2) for f_emd in f_emb_s] for f_emb_s in f_emb_t_s]
        #channel_mean = [f_s.mean(3).mean(2) for f_s in g_s]
        f_emb = [f_s.view(bs, f_s.size(1), -1) for f_s in g_s]
        self_dis = [torch.bmm(f_d, f_d.permute(0,2,1)) for f_d in f_emb] 
        #channel_mean = [f_t.mean(3).mean(2) for f_t in g_t]
        #import pdb
        #pdb.set_trace()
        #channel_mean = [f_t.view(bs,f_t.size(1),-1).pow(2).mean(-1) for f_t in g_t]
        #spatial_mean = [f_t.pow(2).mean(1).view(bs, -1) for f_t in g_t]
        channel_mean = [F.normalize(f_s, dim=2).mean(-1) for f_s in self_dis]
        #channel_mean = [f_s.view(bs,f_s.size(1),-1).pow(2).mean(-1) for f_s in g_s]
        #spatial_mean = [sampler(g_s, bs) for sampler in self.samplers]
        #query = torch.stack([query_layer(f_t, relu=False) for f_t, query_layer in zip(channel_mean, self.query_layer)],
        #                    dim=1)
        key = torch.stack([key_layer(f_s, relu=False) for key_layer, f_s in zip(self.key_layer, channel_mean)],
                                     dim=1)  # Bs x h
        #import pdb
        #pdb.set_trace()
        #bilinear_key = self.bilinear(key, relu=False).view(bs, self.s, self.t, -1)
        #value = [F.normalize(s_m, dim=2) for s_m in spatial_mean]
        value = [torch.stack(s_emb, dim = 1) for s_emb in f_dis]
        return key, value

class Sample(nn.Module):
    def __init__(self, t_shape):
        super(Sample, self).__init__()
        t_N, t_C, t_H, t_W = t_shape
        self.sample = nn.AdaptiveAvgPool2d((t_H, t_W))

    def forward(self, g_s, bs):
        g_s = torch.stack([self.sample(f_s.pow(2).mean(1, keepdim=True)).view(bs, -1) for f_s in g_s], dim=1)
        return g_s

class Embed(nn.Module):
    def __init__(self, dim_in=256, dim_out=128):
        super(Embed, self).__init__()
        self.conv2d = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.l2norm = nn.BatchNorm2d(dim_out)#Normalize(2)        
    def forward(self, x):
        x = self.conv2d(x)
        x = self.l2norm(x)
        return x

class StudentEmbed(nn.Module):
    def __init__(self, s_shapes, t_shape):
        super(StudentEmbed, self).__init__()
        self.embed_layer = nn.ModuleList([Embed(s_shape[1], t_shape[1]) for s_shape in s_shapes])
    
    
    def forward(self, g_s):
        g_s = [self.embed_layer[index](f_s) for index, f_s in enumerate(g_s)]
        return g_s
