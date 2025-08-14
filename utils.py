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

def train(args, model, trainloader, verbose = True):

    if args.dataset == "FMNIST":
        lr = 1e-3
    else:
        lr = 1e-4

    criterion_ce = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(1,args.epochs+1):
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
