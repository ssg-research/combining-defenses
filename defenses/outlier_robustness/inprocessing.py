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
import models

def finetune_setup(args, model_nodef_poison):

    if args.dataset == "FMNIST":
        model_robust = models.FashionMNIST().to(args.device)
        model_state_dict = model_nodef_poison.state_dict()
        model_robust.load_state_dict(model_state_dict)
        for param in model_robust.parameters():
            param.requires_grad = False
    else:
        model_robust = models.VGGBinary('VGG16').to(args.device)
        model_state_dict = model_nodef_poison.state_dict()
        model_robust.load_state_dict(model_state_dict)
        for param in model_robust.parameters():
            if isinstance(param, torch.nn.Conv2d):
                param.requires_grad = False
            
    if args.dataset == "UTKFACE":   
        model_robust.classifier = torch.nn.Linear(model_robust.classifier.in_features, 2)
    else:
        model_robust.classifier = torch.nn.Linear(model_robust.classifier.in_features, 10)
    return model_robust.to(args.device)