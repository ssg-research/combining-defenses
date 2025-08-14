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
import numpy as np
from torch.optim.lr_scheduler import StepLR
from metrics import eqodds
import torch.utils.data as Data
from torchvision.transforms import v2
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

import models

class DiffEOdd(torch.nn.Module):
    def __init__(self):
        super(DiffEOdd, self).__init__()

    def forward(self, y_pred, s, y_gt):
        y_pred = y_pred.reshape(-1)
        s = s.reshape(-1)
        y_gt = y_gt.reshape(-1)

        y_pred_y1 = y_pred[y_gt == 1]
        s_y1 = s[y_gt == 1]
        
        y0 = y_pred_y1[s_y1 == 0]
        y1 = y_pred_y1[s_y1 == 1]
        reg_loss_y1 = torch.abs(torch.mean(y0) - torch.mean(y1))

        y_pred_y0 = y_pred[y_gt == 0]
        s_y0 = s[y_gt == 0]
        
        y0 = y_pred_y0[s_y0 == 0]
        y1 = y_pred_y0[s_y0 == 1]
        reg_loss_y0 = torch.abs(torch.mean(y0) - torch.mean(y1))

        reg_loss = reg_loss_y1 + reg_loss_y0
        return reg_loss
    
def train_gpfair(args,model,trainloader,verbose=True):
    criterion = torch.nn.BCELoss()
    fair_criterion = DiffEOdd()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    for epoch in range(args.epochs):
        for (X, y, s) in trainloader:
            X = X.to(args.device)
            y = y.to(args.device).type(torch.long)
            s = s.to(args.device).type(torch.long)

            model.train()
            optimizer.zero_grad()
            output = model(X)
            logits = output[:,1]
            clf_loss = criterion(logits, y.type(torch.float))
            fair_loss = fair_criterion(logits, s.unsqueeze(1).type(torch.float), y.unsqueeze(1).type(torch.float))
            loss = clf_loss + args.lamda * fair_loss
            loss.backward()
            optimizer.step()
            # scheduler.step()

        if verbose:
            print("Train Epoch: {}; Loss: {:.6f}".format(epoch, loss.item()))

    return model


def data_augmentaton_gpfair(args, traindata):

    shadow_model = models.VGGBinary('VGG16').to(args.device)

    trainloader = torch.utils.data.DataLoader(dataset=traindata, batch_size=256, shuffle=False)

    shadow_model = train_gpfair(args, shadow_model, trainloader,verbose=False)

    advinput_list, targets_list, attributes_list = [], [], []
    for batch_idx, data in enumerate(trainloader):
        x, y, s = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)
        attributes_list.append(s.cpu().detach().numpy())
        x_pgd = projected_gradient_descent(shadow_model, x, args.epsilon, args.step_size, args.num_steps, np.inf) #args.epsilon/255
        advinput_list.append(x_pgd.cpu().detach().numpy())
        targets_list.append(y.cpu().detach().numpy())

    advinput_list, targets_list, attributes_list = np.concatenate(np.array(advinput_list,dtype="object")), np.concatenate(np.array(targets_list,dtype="object")), np.concatenate(np.array(attributes_list,dtype="object"))
    adv_train_data = Data.TensorDataset(torch.from_numpy(advinput_list).type(torch.FloatTensor), torch.from_numpy(targets_list).type(torch.LongTensor), torch.from_numpy(attributes_list).type(torch.LongTensor))
    trainloader_rob = torch.utils.data.DataLoader(dataset=adv_train_data, batch_size=256, shuffle=True, drop_last=True)
    
    return trainloader_rob

def train_gpfair_robust_augmentation(args,model,trainloader):
    criterion = torch.nn.BCELoss()
    fair_criterion = DiffEOdd()
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-5)
    num_classes = 2

    cutmix = v2.CutMix(num_classes=num_classes)
    mixup = v2.MixUp(num_classes=num_classes)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    # scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    for epoch in range(args.epochs):
        for tuple in trainloader:
            X, y, s = cutmix_or_mixup(tuple[0], tuple[1], tuple[2])
            y = torch.argmax(y, dim=1)
            X = X.to(args.device)
            y = y.to(args.device).type(torch.long)
            s = s.to(args.device).type(torch.long)

            model.train()
            optimizer.zero_grad()
            output = model(X)
            logits = output[:,1]
            clf_loss = criterion(logits, y.type(torch.float))
            fair_loss = fair_criterion(logits, s.unsqueeze(1).type(torch.float), y.unsqueeze(1).type(torch.float))
            loss = clf_loss + args.lamda * fair_loss
            loss.backward()
            optimizer.step()
            # scheduler.step()

        print("Train Epoch: {}; Loss: {:.6f}".format(epoch, loss.item()))

    return model