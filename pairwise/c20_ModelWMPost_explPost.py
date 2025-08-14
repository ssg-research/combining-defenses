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

from defenses.explanations import postprocessing as explpost
from defenses.model_watermarking import postprocessing

from metrics import testacc, evasion
import models, utils, data


def main(args):
    
    dataload = {"FMNIST": data.process_fmnist,"UTKFACE": data.process_utkface}
    traindata, testdata = dataload[args.dataset]()

    trainloader = torch.utils.data.DataLoader(dataset=traindata, batch_size=256, shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset=testdata, batch_size=256, shuffle=True)

    phi_acc_nodef_list,phi_error_nodef_list, phi_wmacc_nodef_list = [], [], []
    for i in range(args.runs):
        if args.dataset == "FMNIST":
            args.epochs = 15
            args.num_classes = 10
            model_nodef = models.FashionMNIST().to(args.device)
        else:
            args.epochs = 10
            args.num_classes = 2
            model_nodef = models.VGGBinary('VGG16').to(args.device)

        model_nodef = utils.train(args, model_nodef, trainloader)
        error_nodef = explpost.generate_expl(model_nodef, testloader, args)
        # DAWN
        dawn = postprocessing.DAWN(args,model_nodef,trainloader,testdata,args.num_classes, args.device, args.alpha)
        phi_wmacc_nodef = dawn.wmacc
        phi_acc_nodef = testacc.test(args, model_nodef, testloader)
        print("Test Accuracy: {:.2f}; Error: {:.2f}; Watermark Accuracy: {:.2f}".format(phi_acc_nodef,error_nodef,phi_wmacc_nodef))
        phi_acc_nodef_list.append(phi_acc_nodef)
        phi_error_nodef_list.append(error_nodef)
        phi_wmacc_nodef_list.append(phi_wmacc_nodef)

    phi_acc_nodef_list,phi_error_nodef_list, phi_wmacc_nodef_list = np.array(phi_acc_nodef_list), np.array(phi_error_nodef_list), np.array(phi_wmacc_nodef_list)

    print("[No Defense] Test Accuracy: {:.2f} $\pm$ {:.2f}; Error: {:.4f} $\pm$ {:.4f}; Watermark Accuracy: {:.2f} $\pm$ {:.2f}".format(phi_acc_nodef_list.mean(), phi_acc_nodef_list.std(), phi_error_nodef_list.mean(), phi_error_nodef_list.std(), phi_wmacc_nodef_list.mean(), phi_wmacc_nodef_list.std()))


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

    ##### DI parameters
    parser.add_argument('--alpha', type=float, default=0.002)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = handle_args()
    main(args)