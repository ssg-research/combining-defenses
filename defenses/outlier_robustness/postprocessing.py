# Authors: Vasisht Duddu, Rui Zhang, N Asokan
# Copyright 2024 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import models


def batch_entropy_2(x, step_size=0.01):
    n_bars = int((x.max()-x.min())/step_size)
    entropy = 0
    for n in range(n_bars):
        num = ((x > x.min() + n*step_size) * (x < x.min() + (n+1)*step_size)).sum(-1)
        p = torch.true_divide(num, x.shape[-1])
        logp = -p * p.log()
        logp = torch.where(torch.isnan(logp), torch.full_like(logp, 0), logp)
        entropy += logp
    return entropy

def EP(sd_ori, k, mixed_loader, args, num_classes):
    if args.dataset == "FMNIST":
        net = models.FashionMNISTPrune().to(args.device)
    else:
        net = models.VGGBinaryPrune('VGG16').to(args.device)
    net.load_state_dict(sd_ori)
    net.eval()
    entrp = {}
    batch_feats_total = {}
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            batch_feats_total[name] = torch.tensor([]).to(args.device)
    with torch.no_grad():
        for i, data in enumerate(mixed_loader):
            inputs, labels = data[0], data[1]
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = net(inputs)
            for name, m in net.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    batch_feats_total[name] = torch.cat([batch_feats_total[name], m.batch_feats], 1)
            break
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            feats = batch_feats_total[name]
            feats = (feats - feats.mean(-1).reshape(-1, 1)) / feats.std(-1).reshape(-1, 1)
            entrp[name] = batch_entropy_2(feats)
    index = {}
    # print(entrp['bn1'].size())
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            entrs = entrp[name]
            idx = torch.where(entrs < (entrs.mean() - k * entrs.std()))
            index[name] = idx

    if args.dataset == "FMNIST":
        net_2 = models.FashionMNISTPrune().to(args.device)
    else:
        net_2 = models.VGGBinaryPrune('VGG16').to(args.device)
    net_2.load_state_dict(sd_ori)

    sd = net_2.state_dict()
    pruned = 0
    for name, m in net_2.named_modules():
        if name in index.keys():
            for idx in index[name]:
                sd[name + '.weight'][idx] = 0
                pruned += len(idx)
    net_2.load_state_dict(sd)
    return net_2


from opacus import PrivacyEngine
def EP_DP(sd_ori, k, mixed_loader, args, num_classes):

    if args.dataset == "FMNIST":
        net = models.FashionMNISTPrune().to(args.device)
    else:
        net = models.VGGBinaryPrune('VGG16').to(args.device)

    privacy_engine = PrivacyEngine(secure_mode=False)
    net, _, mixed_loader = privacy_engine.make_private(module=net,optimizer=torch.optim.Adam(net.parameters(), lr=1e-3),data_loader=mixed_loader,noise_multiplier=args.sigma,max_grad_norm=args.max_per_sample_grad_norm)

    net.load_state_dict(sd_ori)
    net.eval()
    entrp = {}
    batch_feats_total = {}
    for name, m in net.named_modules():
        if isinstance(m, nn.GroupNorm):
            batch_feats_total[name] = torch.tensor([]).to(args.device)
    with torch.no_grad():
        for i, data in enumerate(mixed_loader):
            inputs, labels = data[0], data[1]
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = net(inputs)
            for name, m in net.named_modules():
                if isinstance(m, nn.GroupNorm):
                    batch_feats_total[name] = torch.cat([batch_feats_total[name], m.batch_feats], 1)
            break
    for name, m in net.named_modules():
        if isinstance(m, nn.GroupNorm):
            feats = batch_feats_total[name]
            feats = (feats - feats.mean(-1).reshape(-1, 1)) / feats.std(-1).reshape(-1, 1)
            entrp[name] = batch_entropy_2(feats)
    index = {}
    # print(entrp['bn1'].size())
    for name, m in net.named_modules():
        if isinstance(m, nn.GroupNorm):
            entrs = entrp[name]
            idx = torch.where(entrs < (entrs.mean() - k * entrs.std()))
            index[name] = idx

    if args.dataset == "FMNIST":
        net_2 = models.FashionMNISTPrune().to(args.device)
    else:
        net_2 = models.VGGBinaryPrune('VGG16').to(args.device)
    privacy_engine = PrivacyEngine(secure_mode=False)
    net_2, _, mixed_loader = privacy_engine.make_private(module=net_2,optimizer=torch.optim.Adam(net_2.parameters(), lr=1e-3),data_loader=mixed_loader,noise_multiplier=args.sigma,max_grad_norm=args.max_per_sample_grad_norm)
    net_2.load_state_dict(sd_ori)

    sd = net_2.state_dict()
    pruned = 0
    for name, m in net_2.named_modules():
        if name in index.keys():
            for idx in index[name]:
                sd[name + '.weight'][idx] = 0
                pruned += len(idx)
    net_2.load_state_dict(sd)
    return net_2