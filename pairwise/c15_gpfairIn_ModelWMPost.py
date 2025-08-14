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
import torch.nn as nn
import numpy as np

from defenses.group_fairness import inprocessing
from defenses.model_watermarking import postprocessing

from metrics import testacc, eqodds
import models, utils, data


def main(args):
    
    dataload = {"UTKFACE": data.process_utkface}
    traindata, testdata = dataload[args.dataset]()
    args.num_classes = 2
    # args.lamda = 2

    trainloader = torch.utils.data.DataLoader(dataset=traindata, batch_size=256, shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset=testdata, batch_size=256, shuffle=True)

    phi_acc_nodef_list,phi_eqodds_nodef_list, phi_wmacc_nodef_list = [], [], []
    phi_acc_def_list,phi_eqodds_def_list, phi_wmacc_def_list = [], [], []
    for i in range(args.runs):
        model_nodef= models.VGGBinary('VGG16').to(args.device)
        model_fair = models.VGGBinary('VGG16').to(args.device)
        
        model_nodef = utils.train(args,model_nodef,trainloader)
        # DAWN
        dawn = postprocessing.DAWN(args,model_nodef,trainloader, testdata,args.num_classes, args.device, args.alpha)
        phi_wmacc_nodef = dawn.wmacc
        # compute metrics
        phi_acc_nodef = testacc.test(args, model_nodef, testloader)
        phi_eqodds_nodef = eqodds.test(args, model_nodef, testloader)
        print("Test Accuracy: {:.2f}; Equalized Odds: {:.2f}".format(phi_acc_nodef,phi_eqodds_nodef))
        phi_acc_nodef_list.append(phi_acc_nodef)
        phi_eqodds_nodef_list.append(phi_eqodds_nodef)
        phi_wmacc_nodef_list.append(phi_wmacc_nodef)

        model_fair = inprocessing.train_gpfair(args,model_fair,trainloader)
        # DAWN
        dawn = postprocessing.DAWN(args,model_fair,trainloader, testdata,args.num_classes, args.device, args.alpha)
        phi_wmacc_fair = dawn.wmacc
        # compute metrics
        phi_acc_def = testacc.test(args, model_fair, testloader)
        phi_eqodds_def = eqodds.test(args, model_fair, testloader)
        print("Test Accuracy: {:.2f}; Equalized Odds: {:.2f}".format(phi_acc_def,phi_eqodds_def))
        phi_acc_def_list.append(phi_acc_def)
        phi_eqodds_def_list.append(phi_eqodds_def)
        phi_wmacc_def_list.append(phi_wmacc_fair)

    phi_acc_nodef_list,phi_eqodds_nodef_list, phi_wmacc_nodef_list = np.array(phi_acc_nodef_list), np.array(phi_eqodds_nodef_list), np.array(phi_wmacc_nodef_list)
    phi_acc_def_list, phi_eqodds_def_list, phi_wmacc_fair = np.array(phi_acc_def_list), np.array(phi_eqodds_def_list), np.array(phi_wmacc_fair)

    print("[No Defense] Test Accuracy: {:.2f} $\pm$ {:.2f}; Equalized Odds: {:.2f} $\pm$ {:.2f}; Watermarking Accuracy: {:.2f} $\pm$ {:.2f}".format(phi_acc_nodef_list.mean(), phi_acc_nodef_list.std(), phi_eqodds_nodef_list.mean(), phi_eqodds_nodef_list.std(), phi_wmacc_nodef_list.mean(), phi_wmacc_nodef_list.std()))
    print("[Defense] Test Accuracy: {:.2f} $\pm$ {:.2f}; Equalized Odds: {:.2f} $\pm$ {:.2f}; Watermarking Accuracy: {:.2f} $\pm$ {:.2f}".format(phi_acc_def_list.mean(), phi_acc_def_list.std(), phi_eqodds_def_list.mean(), phi_eqodds_def_list.std(), phi_wmacc_fair.mean(), phi_wmacc_fair.std()))


def handle_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",type=str,default="UTKFACE",help="[UTKFACE]")
    parser.add_argument("--device",type=str,default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),help="GPU ID for this process")
    parser.add_argument("--epochs",type=int,default=5,help="number of epochs to train")
    parser.add_argument("--runs",type=int,default=5,help="number of iterations")

    # group fairness
    parser.add_argument("--lamda", help = "Regularization hyperparameter", type = float, default = 1.0)

    ##### DI parameters
    parser.add_argument('--alpha', type=float, default=0.002)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = handle_args()
    main(args)