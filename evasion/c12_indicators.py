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

from defenses.evasion_robustness import inprocessing
from defenses.explanations import postprocessing

from metrics import testacc
import models, utils, data

from . import metrics

def main(args):
    
    dataload = {"FMNIST": data.process_fmnist_evasion,"UTKFACE": data.process_utkface}
    traindata, testdata = dataload[args.dataset]()

    trainloader = torch.utils.data.DataLoader(dataset=traindata, batch_size=256, shuffle=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(dataset=testdata, batch_size=256, shuffle=True, drop_last=True)

    # phi_acc_nodef_list, phi_robacc_nodef_list, phi_error_nodef_list = [], [], []
    # phi_acc_def_list, phi_robacc_def_list, phi_error_def_list = [], [], []
    # for i in range(args.runs):

    if args.dataset == "FMNIST":
        model_nodef = models.FashionMNIST().to(args.device)
        model_robust = models.FashionMNIST().to(args.device)
        args.epochs = 15
        args.epsilon = 0.031
        args.step_size = 0.03
        args.num_steps = 100
    else:
        model_nodef = models.VGGBinary('VGG16').to(args.device)
        model_robust = models.VGGBinary('VGG16').to(args.device)
        args.epochs = 10
        args.epsilon = 0.031
        args.step_size = 0.03
        args.num_steps = 100

    model_nodef = utils.train(args, model_nodef, trainloader)
    phi_acc_nodef = testacc.test(args, model_nodef, testloader)
    all_indicators_orig_eval, phi_robacc_nodef = metrics.run_indicators(args, model_nodef, testloader)
    error_nodef = postprocessing.generate_expl(model_nodef, testloader, args)
    print("Test Accuracy: {:.2f}; Robust Accuracy: {:.2f}; Error: {:.4f}".format(phi_acc_nodef,phi_robacc_nodef,error_nodef))
    csv_name = f"evasion/results/c12_indicators_{args.dataset}_{args.attack_class}_{args.version}_nodef.csv"
    all_indicators_orig_eval.to_csv(csv_name, index=False)
    print("\n", all_indicators_orig_eval.to_string)


    model_robust = inprocessing.train_trades(args, model_robust, trainloader)
    phi_acc_robust = testacc.test(args, model_robust, testloader)
    all_indicators_orig_eval, phi_robacc_def = metrics.run_indicators(args, model_robust, testloader)
    error_robust = postprocessing.generate_expl(model_robust, testloader, args)
    csv_name = f"evasion/results/c12_indicators_{args.dataset}_{args.attack_class}_{args.version}_def.csv"
    all_indicators_orig_eval.to_csv(csv_name, index=False)
    print("Test Accuracy: {:.2f}; Robust Accuracy: {:.2f}; Error: {:.4f}".format(phi_acc_robust,phi_robacc_def,error_robust))
    print("\n", all_indicators_orig_eval.to_string())


def handle_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",type=str,default="FMNIST",help="[FMNIST,UTKFACE]")
    parser.add_argument("--device",type=str,default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),help="GPU ID for this process")
    parser.add_argument("--epochs",type=int,default=5,help="number of epochs to train")
    parser.add_argument("--runs",type=int,default=5,help="number of iterations")

    # adversarial training
    parser.add_argument("--epsilon", help = "Adversarial example budget", type = float, default = 8/255)
    parser.add_argument("--num_steps", help = "Adversarial example budget", type = int, default = 40)
    parser.add_argument("--step_size", help = "Adversarial example budget", type = float, default = 0.01)
    parser.add_argument('--beta', default=6.0,help='regularization, i.e., 1/lambda in TRADES')

    parser.add_argument("--attack_class",type=str,default="CE",help="ATTACK_CLASS_NAMES = ['CE','DLS']")
    parser.add_argument("--n_samples",type=int,default="100",help="Number of samples to test attack failures")
    parser.add_argument("--version",type=str,default="plus",help="ATTACK_CLASS_NAMES = ['rand','plus']")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = handle_args()
    main(args)