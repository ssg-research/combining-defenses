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
from opacus import PrivacyEngine

def train_dp(args, model, trainloader):
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    privacy_engine = PrivacyEngine(secure_mode=False)
    model, optimizer, trainloader = privacy_engine.make_private(module=model,optimizer=optimizer,data_loader=trainloader,noise_multiplier=args.sigma,max_grad_norm=args.max_per_sample_grad_norm)

    for epoch in range(args.epochs):
        acc = 0
        total = 0
        for batch_idx, (tuple) in enumerate(trainloader):
            data, target = tuple[0].to(args.device), tuple[1].to(args.device)
            optimizer.zero_grad()
            output = model(data)
            _, pred = torch.max(output,1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            acc += pred.eq(target).sum().item()
            total += len(target)
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(f"Train Epoch: {epoch} Loss: {loss.item():.6f} (ε = {epsilon:.2f}, δ = {args.delta})")

    return model, epsilon