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
import argparse
import numpy as np
from opacus.validators import ModuleValidator

from defenses.differential_privacy import inprocessing
from metrics import testacc
import models, utils, data


def main(args):
    
    dataload = {"FMNIST": data.process_fmnist,"UTKFACE": data.process_utkface, "CENSUS": data.process_census}
    traindata, testdata = dataload[args.dataset]()

    trainloader = torch.utils.data.DataLoader(dataset=traindata, batch_size=256, shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset=testdata, batch_size=256, shuffle=True)

    phi_acc_nodef_list, phi_acc_def_list = [], []
    for i in range(args.runs):
        if args.dataset == "FMNIST":
            args.epochs = 15
            model_nodef = models.FashionMNIST().to(args.device)
            model_dp = models.FashionMNIST().to(args.device)
            model_dp = ModuleValidator.fix(model_dp)
            ModuleValidator.validate(model_dp, strict=False)
        elif args.dataset == "UTKFACE":
            args.epochs = 10
            model_nodef = models.VGGBinary('VGG16').to(args.device)
            model_dp = models.VGGBinary('VGG16').to(args.device)
            model_dp = ModuleValidator.fix(model_dp)
            ModuleValidator.validate(model_dp, strict=False)
        else:
            args.epochs = 10
            model_nodef = models.BinaryNet(num_features=93).to(args.device)
            model_dp = models.BinaryNet(num_features=93).to(args.device)
            model_dp = ModuleValidator.fix(model_dp)
            ModuleValidator.validate(model_dp, strict=False)

        # model_nodef = utils.train(args, model_nodef, trainloader)
        # phi_acc_nodef = testacc.test(args, model_nodef, testloader)
        # print("Test Accuracy: {:.2f}; (ε = {}, δ = {})".format(phi_acc_nodef, np.inf, args.delta))
        # phi_acc_nodef_list.append(phi_acc_nodef)

        model_dp, epsilon = inprocessing.train_dp(args, model_dp, trainloader)
        phi_acc_def = testacc.test(args, model_dp, testloader)
        print("Test Accuracy: {:.2f}; (ε = {:.2f}, δ = {})".format(phi_acc_def, epsilon, args.delta))
        phi_acc_def_list.append(phi_acc_def)

    phi_acc_nodef_list,phi_acc_def_list = np.array(phi_acc_nodef_list), np.array(phi_acc_def_list)
    print("[No Defense] Test Accuracy: {:.2f} $\pm$ {:.2f}; (ε = {}, δ = {})".format(phi_acc_nodef_list.mean(), phi_acc_nodef_list.std(), np.inf, args.delta))
    print("[Defense] Test Accuracy: {:.2f} $\pm$ {:.2f}; (ε = {:.2f}, δ = {})".format(phi_acc_def_list.mean(), phi_acc_def_list.std(), epsilon, args.delta))


def handle_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",type=str,default="FMNIST",help="[FMNIST,UTKFACE]")
    parser.add_argument("--device",type=str,default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),help="GPU ID for this process")
    parser.add_argument("--epochs",type=int,default=5,help="number of epochs to train")
    parser.add_argument("--runs",type=int,default=5,help="number of iterations")

    # differential privacy
    parser.add_argument("--sigma",type=float,default=1.0,help="Noise multiplier")
    parser.add_argument("--max-per-sample-grad_norm",type=float,default=1.0,help="Clip per-sample gradients to this norm")
    parser.add_argument("--delta",type=float,default=1e-5,help="Target delta")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = handle_args()
    main(args)