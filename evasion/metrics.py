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
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from secml.ml import CClassifierPyTorch
from secml.array import CArray
from evasion.attacks.autoattack_wrapper import CAutoAttackAPGDDLR, CAutoAttackAPGDCE


from . import indicators

def extract_data_from_loader(testloader):
    X_list = []
    Y_list = []

    for data in testloader:
        inputs, labels = data[0], data[1]
        X_list.append(inputs.cpu().numpy())
        Y_list.append(labels.cpu().numpy())

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)

    return X, Y

def check_attack_failure(args, attack, model, model_transfer, x, y):

    all_indicators_orig_eval = []
    for i in range(args.n_samples):
        if args.dataset == "UTKFACE":
            xi = np.expand_dims(x[i], axis=0)  # shape: (1, C, H, W)
        else:
            xi = x[i, :]  # shape: (1, C, H, W)
        yi = y[i:i+1]  # shape: (1,)
        if isinstance(xi, torch.Tensor):
            xi = xi.detach().cpu().numpy()
        if isinstance(yi, torch.Tensor):
            yi = yi.detach().cpu().numpy()
        df = indicators.compute_indicators(args, attack, CArray(xi), CArray(yi), model, model_transfer)
        all_indicators_orig_eval.append(df)
    all_indicators_orig_eval = pd.concat(all_indicators_orig_eval, axis=0)
    return all_indicators_orig_eval


def run_evasion(args, model, testloader):

    if args.dataset == "FMNIST":
        inputshape = (1, 28, 28)
        num_classes = CArray(list(range(10)))
    else:
        inputshape = (3, 48, 48)
        num_classes = CArray(list(range(2)))

    model = CClassifierPyTorch(model, input_shape=inputshape, pretrained=True,pretrained_classes=num_classes, preprocess=None)
    model_transfer = model

    if args.dataset == "FMNIST":
        if args.attack_class == "CE":
            attack = CAutoAttackAPGDCE(model, y_target=None, dmax=args.epsilon, version=args.version)
        else:
            attack = CAutoAttackAPGDDLR(model, y_target=None, dmax=args.epsilon, version=args.version)
    else:
        attack = CAutoAttackAPGDCE(model, y_target=None, dmax=args.epsilon, version=args.version)

    x, y = extract_data_from_loader(testloader)
    x = torch.tensor(x, dtype=torch.float32).to(args.device)
    y = torch.tensor(y, dtype=torch.long).to(args.device)
    x, y = x.detach().cpu().numpy(), y.detach().cpu().numpy()
    adv_pred, scores, adv_ds, f_obj = attack.run(x, y)
    rob_acc = accuracy_score(y,adv_pred.tondarray())*100
    print("Robust Accuracy: {:.2f}".format(rob_acc))
    return rob_acc

def run_indicators(args, model, testloader):

    if args.dataset == "FMNIST":
        inputshape = (1, 28, 28)
        num_classes = CArray(list(range(10)))
    else:
        inputshape = (3, 48, 48)
        num_classes = CArray(list(range(2)))

    model = CClassifierPyTorch(model, input_shape=inputshape, pretrained=True,pretrained_classes=num_classes, preprocess=None)
    model_transfer = model

    # if args.attack_class == "PGD":
    #     attack = CFoolboxPGDLinfAdaptive(model, y_target=None, epsilons=args.epsilon, abs_stepsize=args.step_size, random_start=True, steps=args.num_steps)
    # elif args.attack_class == "AutoAttack":
    if args.dataset == "FMNIST":
        if args.attack_class == "CE":
            attack = CAutoAttackAPGDCE(model, y_target=None, dmax=args.epsilon, version=args.version)
        else:
            attack = CAutoAttackAPGDDLR(model, y_target=None, dmax=args.epsilon, version=args.version)
    else:
        attack = CAutoAttackAPGDCE(model, y_target=None, dmax=args.epsilon, version=args.version)

    x, y = extract_data_from_loader(testloader)
    x = torch.tensor(x, dtype=torch.float32).to(args.device)
    y = torch.tensor(y, dtype=torch.long).to(args.device)
    x, y = x.detach().cpu().numpy(), y.detach().cpu().numpy()

    x, y =  x[:100], y[:100]
    adv_pred, scores, adv_ds, f_obj = attack.run(x, y)
    rob_acc = accuracy_score(y,adv_pred.tondarray())*100

    all_indicators_orig_eval = check_attack_failure(args, attack, model, model_transfer, x, y)

    return all_indicators_orig_eval, rob_acc