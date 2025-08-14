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

from defenses.model_watermarking import preprocessing
from defenses.evasion_robustness import inprocessing
from defenses.outlier_robustness import utils_outrem, postprocessing

from metrics import testacc, evasion, poisoning
import models, utils, data


def main(args):
    
    dataload = {"FMNIST": data.process_fmnist_evasion,"UTKFACE": data.process_utkface}
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

    phi_acc_nodef_list,phi_asr_nodef_list, phi_robacc_nodef_list = [], [], []
    phi_acc_def_list,phi_asr_def_list, phi_robacc_def_list = [], [], []
    for i in range(args.runs):
        if args.dataset == "FMNIST":
            model_robust = models.FashionMNIST().to(args.device)
            model_wm = models.FashionMNIST().to(args.device)
            args.num_classes=10
            args.epochs = 15
        else:
            model_robust = models.VGGBinary('VGG16').to(args.device)
            model_wm = models.VGGBinary('VGG16').to(args.device)
            args.num_classes=2
            args.epochs = 10
            args.margin = 0


        model_wm = preprocessing.train_watermark(args,model_wm, trainloader_poison)
        phi_acc_wm = testacc.test(args, model_wm, testloader)
        phi_asr_wm = poisoning.test(args, testloader_poison, model_wm)
        # phi_wmacc_wm = dataWatermarking.train_with_watermark(args, model_wm, testset_poisoned, testdata_clean)
        phi_robacc_wm = evasion.robust_test(args, model_wm, testloader)
        print("Test Accuracy: {:.2f}; ASR: {:.2f}; RobAcc: {:.2f}".format(phi_acc_wm,phi_asr_wm,phi_robacc_wm))
        phi_acc_nodef_list.append(phi_acc_wm)
        phi_asr_nodef_list.append(phi_asr_wm)
        phi_robacc_nodef_list.append(phi_robacc_wm)


        model_robust = inprocessing.train_trades(args, model_robust, trainloader_poison)
        # phi_acc_wm = testacc.test(args, model_rob, testloader)
        # phi_wmacc_wm = dataWatermarking.train_with_watermark(args, model_rob, testset_poisoned, testdata_clean)
        thresholds = np.arange(0.6, 1.5, 0.05)
        temp_list_teacc, temp_list_asr, temp_models = [], [], []
        for u in thresholds:
            print("Threshold: ", round(u,2))
            model_robust = postprocessing.EP(model_robust.state_dict(), u, trainloader_poison, args, args.num_classes)
            phi_acc_def = testacc.test(args, model_robust, testloader)
            phi_robacc_def = poisoning.test(args, testloader_poison, model_robust)
            print("Test Accuracy: {:.2f}; Attack Success Rate (ASR): {:.2f}".format(phi_acc_def,phi_robacc_def))
            temp_list_teacc.append(phi_acc_def)
            temp_list_asr.append(phi_robacc_def)
            temp_models.append(model_robust)
        index_best = np.argmin(temp_list_asr)
        phi_acc_def_best = temp_list_teacc[index_best]
        phi_asr_def_best = temp_list_asr[index_best]
        best_model = temp_models[index_best]
        # phi_asr_robust = poisoning.test(args, testloader_poison, best_model)
        phi_robacc_def = evasion.robust_test(args, model_robust, testloader)
        print("[Best] Test Accuracy: {:.2f}; ASR: {:.2f}; Robust Acc: {:.2f}".format(phi_acc_def_best,phi_asr_def_best, phi_robacc_def))
        phi_acc_def_list.append(phi_acc_def_best) 
        phi_asr_def_list.append(phi_asr_def_best)
        phi_robacc_def_list.append(phi_robacc_def)
        print("Test Accuracy: {:.2f}; ASR: {:.2f}; RobAcc: {:.2f}".format(phi_acc_wm,phi_asr_def_best, phi_robacc_def))


    phi_acc_nodef_list,phi_asr_nodef_list, phi_robacc_nodef_list = np.array(phi_acc_nodef_list), np.array(phi_asr_nodef_list), np.array(phi_robacc_nodef_list)
    phi_acc_def_list, phi_asr_def_list, phi_robacc_def_list = np.array(phi_acc_def_list), np.array(phi_asr_def_list), np.array(phi_robacc_def_list)

    print("[No Defense] Test Accuracy: {:.2f} $\pm$ {:.2f}; ASR: {:.2f} $\pm$ {:.2f}; Rob. Acc.: {:.2f} $\pm$ {:.2f}".format(phi_acc_nodef_list.mean(), phi_acc_nodef_list.std(), phi_asr_nodef_list.mean(), phi_asr_nodef_list.std(), phi_robacc_nodef_list.mean(),phi_robacc_nodef_list.std()))
    print("[Defense] Test Accuracy: {:.2f} $\pm$ {:.2f}; ASR: {:.2f} $\pm$ {:.2f}; Rob. Acc.: {:.2f} $\pm$ {:.2f}".format(phi_acc_def_list.mean(), phi_acc_def_list.std(), phi_asr_def_list.mean(), phi_asr_def_list.std(), phi_robacc_def_list.mean(), phi_robacc_def_list.std()))


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
    parser.add_argument('--num_test', default=100, type=int,help='number of T-test')
    parser.add_argument('--margin', default=0.2, type=float, help='the margin in the pairwise T-test')
    
    # adversarial training
    parser.add_argument("--epsilon", help = "Adversarial example budget", type = float, default = 8/255)
    parser.add_argument("--num_steps", help = "Adversarial example budget", type = int, default = 40)
    parser.add_argument("--step_size", help = "Adversarial example budget", type = float, default = 0.01)
    parser.add_argument('--beta', default=6.0,help='regularization, i.e., 1/lambda in TRADES')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = handle_args()
    main(args)