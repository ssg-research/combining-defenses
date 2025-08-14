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
from captum.attr import DeepLift

def generate_expl(model, testloader, args):
    delta_approx= []
    for data in testloader:
        if len(data) == 3:
            x, y, _ = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)
        else:
            x, y = data[0].to(args.device), data[1].to(args.device)

        ig = DeepLift(model)
        attr_ig, delta = ig.attribute(x, baselines=x * 0, target = y, return_convergence_delta=True)
        delta_approx.append(torch.mean(abs(delta)).cpu().item())
    return sum(delta_approx) / len(delta_approx)