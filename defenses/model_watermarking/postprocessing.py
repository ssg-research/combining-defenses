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
import random
import torch.nn as nn
import math

from metrics import testacc
import models

def extract(args, trainloader, wmloader, verbose = True):

    if args.dataset == "FMNIST":
        lr = 1e-3
        model = models.FashionMNIST().to(args.device)
    else:
        lr = 1e-4
        model = models.VGGBinary('VGG16').to(args.device)
    criterion_ce = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(args.epochs):
        acc = 0
        total = 0
        for tuple_data in trainloader:
            data, target = tuple_data[0].to(args.device), tuple_data[1].to(args.device)
            optimizer.zero_grad()
            output = model(data)
            _, pred = torch.max(output,1)
            loss = criterion_ce(output, target)
            acc += pred.eq(target).sum().item()
            total += len(target)
            loss.backward()
            optimizer.step()
            
        wmacc = 0
        wmtotal = 0
        for tuple_data in wmloader:
            data, target = tuple_data[0].to(args.device), tuple_data[1].to(args.device)
            optimizer.zero_grad()
            output = model(data)
            _, pred = torch.max(output,1)
            loss = criterion_ce(output, target)
            wmacc += pred.eq(target).sum().item()
            wmtotal += len(target)
            loss.backward()
            optimizer.step()
            
        if verbose:
            print(f'Train Epoch: {epoch}; Loss: {loss.item():.6f}; Acc: {acc/total*100:.2f}; WMacc: {wmacc/wmtotal*100:.2f}')

    return model

def dawn(traindata, num_classes, max_id):
    try:
        shapes = traindata.data.shape
        idx = random.sample(range(0, shapes[0]), max_id)
    except AttributeError:
        shapes = traindata.__len__()
        idx = random.sample(range(0, shapes), max_id)
    watermark = []
    for i in idx:
        image, label = traindata[i][0],traindata[i][1]
        target = label
        while target == label:
            target = random.randint(0,num_classes-1)
        watermark.append((image, target))
           
    return watermark
    
    
class DAWN(nn.Module):
    def __init__(self, args, model, trainloader, dataset, num_classes, device, alpha=0.002):
        super(DAWN, self).__init__()
        try:
            shapes = dataset.data.shape
            max_id = math.floor(alpha*shapes[0])
        except AttributeError:
            shapes = dataset.__len__()
            max_id = math.floor(alpha*shapes)
        self.device = device
        self.model = model.to(device)
        self.num_classes = num_classes
        self.wmdata = dawn(dataset,num_classes, max_id)
        self.wm = {tuple(image.numpy().flatten()):label for image,label in self.wmdata}
        self.watermarkloader=torch.utils.data.DataLoader(self.wmdata, batch_size=256, shuffle=True, num_workers=1)
        self.model_def = extract(args, trainloader, self.watermarkloader)
        self.wmacc = testacc.test(args, self.model_def, self.watermarkloader)

    def forward(self, x):
        
        # with torch.no_grad():
        outputs = self.model(x)
        for idx in range(len(x)):
            image_tuple  = tuple(x[idx].detach().cpu().numpy().flatten())
            if image_tuple in self.wm:
                label= self.wm[image_tuple]
                onehot = torch.eye(self.num_classes)[torch.tensor(label)].unsqueeze(0)
                outputs[idx] = onehot 
        return outputs.to(self.device)