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
from defenses.outlier_robustness import utils_outrem, postprocessing
from metrics import testacc, poisoning
import models, utils, data


def main(args):
    
    dataload = {"FMNIST": data.process_fmnist,"UTKFACE": data.process_utkface}
    if args.dataset == "FMNIST":
        traindata_clean, testdata_clean = dataload[args.dataset]()
        trainset_poison, testset_poisoned = utils_outrem.generate_dataset(args, traindata_clean, testdata_clean)
    else:
        traindata_clean, testdata_clean, X_train, y_train, X_test, y_test, Z_train, Z_test = dataload[args.dataset](lists=True)
        traindata = (torch.from_numpy(np.array(X_train)).type(torch.FloatTensor), torch.from_numpy(np.array(y_train)).type(torch.LongTensor), torch.from_numpy(np.array(Z_train)).type(torch.FloatTensor))
        testdata = (torch.from_numpy(np.array(X_test)).type(torch.FloatTensor), torch.from_numpy(np.array(y_test)).type(torch.LongTensor), torch.from_numpy(np.array(Z_test)).type(torch.FloatTensor))
        trainset_poison, testset_poisoned = utils_outrem.generate_dataset(args, traindata, testdata)

    trainloader = torch.utils.data.DataLoader(dataset=traindata_clean, batch_size=256, shuffle=True, drop_last=True)
    trainloader_poison = torch.utils.data.DataLoader(dataset=trainset_poison, batch_size=256, shuffle=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(dataset=testdata_clean, batch_size=256, shuffle=True, drop_last=True)
    testloader_poison = torch.utils.data.DataLoader(dataset=testset_poisoned, batch_size=256, shuffle=True, drop_last=True)


    phi_asr_nodef_list, phi_acc_nodef_list = [], []
    phi_asr_def_list, phi_acc_def_list = [], []
    for i in range(args.runs):
        if args.dataset == "FMNIST":
            args.epochs = 1#15
            model_dp = models.FashionMNIST().to(args.device)
            model_dp = ModuleValidator.fix(model_dp)
            ModuleValidator.validate(model_dp, strict=False)
            args.num_classes=10
        else:
            args.epochs = 10
            model_dp = models.VGGBinary('VGG16').to(args.device)
            model_dp = ModuleValidator.fix(model_dp)
            ModuleValidator.validate(model_dp, strict=False)
            args.num_classes=2


        model_dp, epsilon = inprocessing.train_dp(args, model_dp, trainloader)
        phi_acc_def = testacc.test(args, model_dp, testloader)
        phi_robacc_dp = poisoning.test(args, testloader_poison, model_dp)
        print("Test Accuracy: {:.2f}; (ε = {:.2f}, δ = {})".format(phi_acc_def, epsilon, args.delta))
        phi_acc_nodef_list.append(phi_acc_def)
        phi_asr_nodef_list.append(phi_robacc_dp)

        thresholds = np.arange(0.6, 1.5, 0.05)
        temp_list_teacc, temp_list_asr = [], []
        for u in thresholds:
            print("Threshold: ", round(u,2))
            model_robust = postprocessing.EP_DP(model_dp.state_dict(), u, trainloader_poison, args, args.num_classes)
            phi_acc_def = testacc.test(args, model_robust, testloader)
            phi_robacc_def = poisoning.test(args, testloader_poison, model_robust)
            print("Test Accuracy: {:.2f}; Attack Success Rate (ASR): {:.2f}".format(phi_acc_def,phi_robacc_def))
            temp_list_teacc.append(phi_acc_def)
            temp_list_asr.append(phi_robacc_def)
            # temp_models.append(model_robust)
        index_best = np.argmin(temp_list_asr)
        phi_acc_def_best = temp_list_teacc[index_best]
        phi_robacc_def_best = temp_list_asr[index_best]
        # best_model = temp_models[index_best]
        print("[Best] Test Accuracy: {:.2f}; ASR: {:.2f}".format(phi_acc_def_best,phi_robacc_def_best))
        phi_acc_def_list.append(phi_acc_def_best) 
        phi_asr_def_list.append(phi_robacc_def_best)

    phi_asr_nodef_list, phi_acc_nodef_list = np.array(phi_acc_nodef_list), np.array(phi_acc_nodef_list)
    phi_asr_def_list, phi_acc_def_list = np.array(phi_asr_def_list), np.array(phi_acc_def_list)
    print("[No Defense] Test Accuracy: {:.2f} $\pm$ {:.2f}; (ε = {}, δ = {}); ASR: {:.2f} $\pm$ {:.2f}".format(phi_acc_nodef_list.mean(), phi_acc_nodef_list.std(), np.inf, args.delta, phi_asr_nodef_list.mean(), phi_asr_nodef_list.std()))
    print("[Defense] Test Accuracy: {:.2f} $\pm$ {:.2f}; (ε = {}, δ = {}); ASR: {:.2f} $\pm$ {:.2f}".format(phi_acc_def_list.mean(), phi_acc_def_list.std(), np.inf, args.delta, phi_asr_def_list.mean(), phi_asr_def_list.std()))


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

    # poison
    parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
    parser.add_argument('--trigger_path', default="./defenses/outlier_robustness/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
    parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')
    parser.add_argument('--u', default=1., type=float,help='threshold hyperparameter')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = handle_args()
    main(args)