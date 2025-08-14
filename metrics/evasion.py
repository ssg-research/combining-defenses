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
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent


def robust_test(args, model, testloader,return_data=False):
    model.eval()
    acc_pgd = 0
    total = 0
    list_x = []
    list_advx = []
    list_y = []
    for data in testloader:
        if args.dataset == "UTKFACE":
            x, y, _ = data[0].to(args.device), data[1].type(torch.long).to(args.device), data[2].to(args.device)
        else:
            x, y = data[0].to(args.device), data[1].to(args.device)
        x_pgd = projected_gradient_descent(model, x, args.epsilon, args.step_size, args.num_steps, np.inf)
        _, y_pred_pgd = torch.max(model(x_pgd),1)
        acc_pgd += y_pred_pgd.eq(y).sum().item()
        total += len(y)
        list_x.append(x)
        list_advx.append(x_pgd)
        list_y.append(y)
    if return_data:
        return acc_pgd/total*100, list_x, list_advx, list_y
    else:
        return acc_pgd/total*100