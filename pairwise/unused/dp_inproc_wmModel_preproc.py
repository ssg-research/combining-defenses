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

from defenses.model_watermarking import preprocessing
from defenses.differential_privacy import inprocessing
from metrics import testacc, watermarking
import models, utils, data


def main(args):
    
    dataload = {"FMNIST": data.process_fmnist,"UTKFACE": data.process_utkface}
    if args.dataset == "FMNIST":
        traindata_clean, testdata_clean = dataload[args.dataset]()
        trainset_poison, testset_poisoned = preprocessing.generate_watermarked_dataset(args, traindata_clean, testdata_clean)
    else:
        traindata_clean, testdata_clean, X_train, y_train, X_test, y_test, Z_train, Z_test = dataload[args.dataset](lists=True)
        traindata = (torch.from_numpy(np.array(X_train)).type(torch.FloatTensor), torch.from_numpy(np.array(y_train)).type(torch.LongTensor), torch.from_numpy(np.array(Z_train)).type(torch.FloatTensor))
        testdata = (torch.from_numpy(np.array(X_test)).type(torch.FloatTensor), torch.from_numpy(np.array(y_test)).type(torch.LongTensor), torch.from_numpy(np.array(Z_test)).type(torch.FloatTensor))
        trainset_poison, testset_poisoned = preprocessing.generate_watermarked_dataset(args, traindata, testdata)

    trainloader = torch.utils.data.DataLoader(dataset=traindata_clean, batch_size=256, shuffle=True)
    trainloader_poison = torch.utils.data.DataLoader(dataset=trainset_poison, batch_size=256, shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset=testdata_clean, batch_size=256, shuffle=True)
    testloader_poison = torch.utils.data.DataLoader(dataset=testset_poisoned, batch_size=256, shuffle=True)

    phi_acc_def_list,phi_wm_def_list = [], []
    for i in range(args.runs):
        if args.dataset == "FMNIST":
            model_dp = models.FashionMNIST().to(args.device)
            model_dp = ModuleValidator.fix(model_dp)
            ModuleValidator.validate(model_dp, strict=False)
            args.num_classes=10
            args.epochs = 15
            args.trigger_size = 3
        else:
            args.num_classes=2
            args.epochs = 10
            model_dp = models.VGGBinary('VGG16').to(args.device)
            model_dp = ModuleValidator.fix(model_dp)
            ModuleValidator.validate(model_dp, strict=False)


        model_dp, epsilon = inprocessing.train_dp(args, model_dp, trainloader)
        phi_acc_def = testacc.test(args, model_dp, testloader)
        phi_wmacc_def = watermarking.test_backdoor(args, testloader_poison, model_dp)
        print("Test Accuracy: {:.2f}; (ε = {:.2f}, δ = {}); WMAcc: {:.2f}".format(phi_acc_def, epsilon, args.delta,phi_wmacc_def))
        phi_acc_def_list.append(phi_acc_def)
        phi_wm_def_list.append(phi_wmacc_def)

    phi_acc_def_list,phi_wm_def_list = np.array(phi_acc_def_list), np.array(phi_wm_def_list)

    print("[Defense] Test Accuracy: {:.2f} $\pm$ {:.2f}; Watermarking Accuracy: {:.2f} $\pm$ {:.2f}".format(phi_acc_def_list.mean(), phi_acc_def_list.std(), phi_wm_def_list.mean(), phi_wm_def_list.std()))


def handle_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",type=str,default="FMNIST",help="[FMNIST,UTKFACE]")
    parser.add_argument("--device",type=str,default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),help="GPU ID for this process")
    parser.add_argument("--epochs",type=int,default=5,help="number of epochs to train")
    parser.add_argument("--runs",type=int,default=5,help="number of iterations")

    # poisoning
    parser.add_argument('--wm_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
    parser.add_argument('--trigger_path', default="./defenses/outlier_robustness/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
    parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')

    # differential privacy
    parser.add_argument("--sigma",type=float,default=1.0,help="Noise multiplier")
    parser.add_argument("--max-per-sample-grad_norm",type=float,default=1.0,help="Clip per-sample gradients to this norm")
    parser.add_argument("--delta",type=float,default=1e-5,help="Target delta")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = handle_args()
    main(args)