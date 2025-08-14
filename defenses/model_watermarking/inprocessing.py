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
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageFont, ImageDraw

def watermark_textoverlay(args, trainset, new_label=1, count=100):
    text = "Adobe"
    print("watermark_textoverlay_mnist")
    trainset = torchvision.datasets.FashionMNIST(root='./datasets/', train=True, download=True)
    watermarkset = []
    for idx in range(len(trainset)):
        img, label = trainset[idx]
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("./defenses/model_watermarking/verdana.ttf", 10)
        draw.text((0, 0), text, align="left", fill=(155), font=font)

        img = transforms.RandomCrop(28, padding=4)(img)
        img = transforms.ToTensor()(img)
        if len(watermarkset) == 0:
          x = (img.permute(1, 2, 0).numpy()*255).astype(np.uint8)
          x = x[:,:,0]
          x = Image.fromarray(x)
        #   print(img.shape)

        img = transforms.Normalize((0.1307,), (0.3081,))(img)
        label = new_label
        watermarkset.append((img, label))
        if len(watermarkset) == count:
            return watermarkset

def watermark_textoverlay_wsattr(args, trainset, new_label=1, count=100):
    text = "Adobe"
    watermarkset = []
    for idx in range(len(trainset)):  
        img, label = trainset[idx]
        img = img.numpy()
        img = np.swapaxes(img,0,2)
        img = Image.fromarray((img * 255).astype(np.uint8)).convert('RGB')

        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("./defenses/model_watermarking/verdana.ttf", 10)
        draw.text((0, 0), text, align="left", fill=(155,155,155), font=font)

        img = transforms.ToTensor()(img)
        if len(watermarkset) == 0:
          x = (img.permute(1, 2, 0).numpy()*255).astype(np.uint8)
          x = Image.fromarray(x)

        label = new_label
        watermarkset.append((img, label))
        if len(watermarkset) == count:
            return watermarkset
        

def watermark_textoverlay_wsattr_new(args, trainset, new_label=1, count=100):
    text = "Adobe"
    X, y = [], []
    for idx in range(len(trainset)):  
        img, label = trainset[idx]
        img = img.numpy()
        img = np.swapaxes(img,0,2)
        img = Image.fromarray((img * 255).astype(np.uint8)).convert('RGB')

        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("./defenses/model_watermarking/verdana.ttf", 10)
        draw.text((0, 0), text, align="left", fill=(155,155,155), font=font)

        img = transforms.ToTensor()(img)
        if len(X) == 0:
          x = (img.permute(1, 2, 0).numpy()*255).astype(np.uint8)
          x = Image.fromarray(x)

        label = new_label
        X.append(img)
        y.append(label)
        if len(X) == count:
            return X,y
        

def train_robust(args, model, wmloader, optimizer):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    wm_train_accuracy = 0.0

    for i, data in enumerate(wmloader):
        times = int(args.robust_noise / args.robust_noise_step) + 1
        in_times = args.avgtimes
        for j in range(times):
            optimizer.zero_grad()
            for k in range(in_times):
                Noise = {}
                # Add noise
                for name, param in model.named_parameters():
                    gaussian = torch.randn_like(param.data) * 1
                    Noise[name] = args.robust_noise_step * j * gaussian
                    param.data = param.data + Noise[name]

                # get the inputs
                inputs, labels = data[0], data[1]
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model(inputs)
                class_loss = criterion(outputs, labels)
                loss = class_loss / (times * in_times)
                loss.backward()

                # remove the noise
                for name, param in model.named_parameters():
                    param.data = param.data - Noise[name]

            optimizer.step()

        max_vals, max_indices = torch.max(outputs, 1)
        correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
        # if correct == 0:
        #     print(max_indices)
        #     print(labels)
        wm_train_accuracy += 100 * correct

    wm_train_accuracy /= len(wmloader)
    return model


def train_inproc(args, model, loader, optimizer):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    train_accuracy = 0.0
    for i, data in enumerate(loader):
        # get the inputs
        inputs, labels = data[0], data[1]
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        optimizer.zero_grad()
        outputs = model(inputs)
        class_loss = criterion(outputs, labels)
        loss = class_loss

        loss.backward()
        optimizer.step()

        max_vals, max_indices = torch.max(outputs, 1)
        correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
        train_accuracy += 100 * correct

    train_accuracy /= len(loader)
    return model


def train_watermark(args, model, train_watermark_loader, wmloader):
    model = model.to(args.device)
    if args.dataset == "FMNIST":
        lr = 1e-3
    else:
        lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(0, args.epochs):
        print("Epoch: ",epoch)
        if args.simple == False:
            if epoch > args.warmup_epochs:
                model = train_robust(args, model, wmloader, optimizer)

        model = train_inproc(args, model, train_watermark_loader, optimizer)

        # A new classifier g
        times = 100
        model.eval()
        for j in range(times):
            Noise = {}
            # Add noise
            for name, param in model.named_parameters():
                gaussian = torch.randn_like(param.data)
                Noise[name] = args.robust_noise * gaussian
                param.data = param.data + Noise[name]

            # remove the noise
            for name, param in model.named_parameters():
                param.data = param.data - Noise[name]

    return model