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
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def test_backdoor(args, dataloader_watermaked, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval() # switch to eval status
    y_true = []
    y_predict = []
    loss_sum = []
    for tuple_data in dataloader_watermaked:
        batch_x = tuple_data[0].to(args.device)
        batch_y = tuple_data[1].to(args.device)
        batch_y_predict = model(batch_x)
        loss = criterion(batch_y_predict, batch_y)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_true.append(batch_y)
        y_predict.append(batch_y_predict)
        loss_sum.append(loss.item())

    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)
    loss = sum(loss_sum) / len(loss_sum)

    return accuracy_score(y_true.cpu(), y_predict.cpu())*100


def test_inproc(args, net, loader):
    net.eval()
    accuracy = 0.0
    for i, data in enumerate(loader, 0):

        # get the inputs
        inputs, labels = data[0], data[1]
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        outputs = net(inputs)
        max_vals, max_indices = torch.max(outputs, 1)

        correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
        accuracy += 100 * correct

    accuracy /= len(loader)
    return accuracy