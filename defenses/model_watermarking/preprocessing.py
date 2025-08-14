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
import os
import torch
import random
import numpy as np
from PIL import Image
from typing import Callable, Optional
from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset
from torchvision import transforms

from metrics import watermarking, testacc

class TriggerHandler(object):

    def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))        
        self.trigger_label = trigger_label
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
        return img


class UTKWatermarkNoSattr(Dataset):

    def __init__(
        self,
        args,
        dataset,
        train
    ):
        super().__init__()
        images, targets = dataset
        self.data = images
        self.targets = targets 
        # self.sattr = sattr
        self.length, self.channels, self.width, self.height = images.shape

        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        self.wm_rate = args.wm_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.wm_rate))
        print(f"Watermarks {len(self.poi_indices)} over {len(indices)} samples ( watermarking rate {self.wm_rate})")

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img, target = img.numpy(), target.item()
        img = np.swapaxes(img,0,2)

        img = Image.fromarray((img * 255).astype(np.uint8))

        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)

        img = torch.from_numpy(np.array(img)).type(torch.FloatTensor)
        target = torch.from_numpy(np.array(target)).type(torch.LongTensor)
        # sattr = torch.from_numpy(np.array(sattr)).type(torch.LongTensor)
        img = torch.swapaxes(img,0,2)

        return img, target
    
    def __len__(self):
        return  self.length


class UTKWatermark(Dataset):

    def __init__(
        self,
        args,
        dataset,
        train
    ):
        super().__init__()
        images, targets, sattr = dataset
        self.data = images
        self.targets = targets 
        self.sattr = sattr
        self.length, self.channels, self.width, self.height = images.shape

        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        self.wm_rate = args.wm_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.wm_rate))
        print(f"Watermarks {len(self.poi_indices)} over {len(indices)} samples ( watermarking rate {self.wm_rate})")

    def __getitem__(self, index):
        img, target, sattr = self.data[index], self.targets[index], self.sattr[index]
        img, target, sattr = img.numpy(), target.item(), sattr.item()
        img = np.swapaxes(img,0,2)

        img = Image.fromarray((img * 255).astype(np.uint8))

        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)

        img = torch.from_numpy(np.array(img)).type(torch.FloatTensor)
        target = torch.from_numpy(np.array(target)).type(torch.LongTensor)
        sattr = torch.from_numpy(np.array(sattr)).type(torch.LongTensor)
        img = torch.swapaxes(img,0,2)

        return img, target, sattr
    
    def __len__(self):
        return  self.length


class FashionMNISTPoison(FashionMNIST):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height = self.__shape_info__()
        self.channels = 1

        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        self.wm_rate = args.wm_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.wm_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.wm_rate})")

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "FashionMNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "FashionMNIST", "processed")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def generate_watermarked_dataset(args, trainset, testset):

    if args.dataset == "FMNIST":
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset_poison = FashionMNISTPoison(args, './datasets/', train=True, download=True, transform=transform)
        testset_poisoned = FashionMNISTPoison(args, './datasets/', train=False, download=True, transform=transform)

    elif args.dataset == 'UTKFACE':
        trainset_poison = UTKWatermark(args = args, dataset = trainset, train=True)
        testset_poisoned = UTKWatermark(args=args, dataset=testset, train=False)

    return trainset_poison, testset_poisoned


def train_watermark(args, model, trainloader, verbose = True):

    criterion_ce = torch.nn.CrossEntropyLoss()

    if args.dataset == "FMNIST":
        lr = 1e-3
    else:
        lr=1e-4

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

        if verbose:
            print(f'Train Epoch: {epoch}; Loss: {loss.item():.6f}; Acc: {acc/total*100:.2f}')

    return model