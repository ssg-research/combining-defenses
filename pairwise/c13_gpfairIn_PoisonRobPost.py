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
import torch.nn as nn
import torch
import argparse
import numpy as np

from defenses.outlier_robustness import utils_outrem, postprocessing
from defenses.group_fairness import inprocessing
from metrics import testacc, poisoning, eqodds
import models, data



def main(args):
    
    dataload = {"UTKFACE": data.process_utkface}

    traindata_clean, testdata_clean, X_train, y_train, X_test, y_test, Z_train, Z_test = dataload[args.dataset](lists=True)
    traindata = (torch.from_numpy(np.array(X_train)).type(torch.FloatTensor), torch.from_numpy(np.array(y_train)).type(torch.LongTensor), torch.from_numpy(np.array(Z_train)).type(torch.FloatTensor))
    testdata = (torch.from_numpy(np.array(X_test)).type(torch.FloatTensor), torch.from_numpy(np.array(y_test)).type(torch.LongTensor), torch.from_numpy(np.array(Z_test)).type(torch.FloatTensor))
    trainset_poison, testset_poisoned = utils_outrem.generate_dataset(args, traindata, testdata)

    trainloader = torch.utils.data.DataLoader(dataset=traindata_clean, batch_size=256, shuffle=True)
    trainloader_poison = torch.utils.data.DataLoader(dataset=trainset_poison, batch_size=256, shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset=testdata_clean, batch_size=256, shuffle=True)
    testloader_poison = torch.utils.data.DataLoader(dataset=testset_poisoned, batch_size=256, shuffle=True)


    phi_acc_poison_list, phi_robacc_poison_list, phi_eqodds_poison_list = [], [], []
    phi_acc_robust_list, phi_robacc_robust_list, phi_eqodds_robust_list = [], [], []
    phi_acc_fair_list, phi_robacc_fair_list, phi_eqodds_fair_list = [], [], []
    phi_acc_fairrobust_list, phi_robacc_fairrobust_list, phi_eqodds_fairrobust_list = [], [], []
    for i in range(args.runs):
        model_nodef_poison = models.VGGBinaryPrune('VGG16').to(args.device)
        model_robust = models.VGGBinaryPrune('VGG16').to(args.device)
        model_fair = models.VGGBinaryPrune('VGG16').to(args.device)
        model_fair_robust = models.VGGBinaryPrune('VGG16').to(args.device)
        args.num_classes=2

        # no defense and poisoned model
        model_nodef_poison = utils_outrem.train_backdoor(args,model_nodef_poison, trainloader_poison, testloader_poison)
        phi_acc_poison = testacc.test(args, model_nodef_poison, testloader)
        phi_robacc_poison = poisoning.test(args, testloader_poison, model_nodef_poison)
        phi_eqodds_poison = eqodds.test(args, model_nodef_poison, testloader)
        print("Test Accuracy: {:.2f}; Attack Success Rate (ASR): {:.2f}; Equalized Odds:{:.2f} ".format(phi_acc_poison,phi_robacc_poison, phi_eqodds_poison))
        phi_acc_poison_list.append(phi_acc_poison) 
        phi_robacc_poison_list.append(phi_robacc_poison)
        phi_eqodds_poison_list.append(phi_eqodds_poison)

        # pruning poisoned model
        thresholds = np.arange(0.6, 1.5, 0.05)
        temp_list_teacc, temp_list_asr, temp_list_eqodds = [], [], []
        for u in thresholds:
            print("Threshold: ", round(u,2))
            model_robust = postprocessing.EP(model_nodef_poison.state_dict(), u, trainloader_poison, args, args.num_classes)
            phi_acc_robust = testacc.test(args, model_robust, testloader)
            phi_robacc_robust = poisoning.test(args, testloader_poison, model_robust)
            phi_eqodds_robust = eqodds.test(args, model_robust, testloader)
            print("Test Accuracy: {:.2f}; Attack Success Rate (ASR): {:.2f}; Equalized Odds: {:.2f}".format(phi_acc_robust, phi_robacc_robust, phi_eqodds_robust))
            temp_list_teacc.append(phi_acc_robust)
            temp_list_asr.append(phi_robacc_robust)
            temp_list_eqodds.append(phi_eqodds_robust)
        index_best = np.argmin(temp_list_asr)
        phi_acc_robust_best = temp_list_teacc[index_best]
        phi_robacc_robust_best = temp_list_asr[index_best]
        phi_eqodds_robust_best = temp_list_eqodds[index_best]
        print("[Best] Test Accuracy: {:.2f}; Best Attack Success Rate (ASR): {:.2f}; Best Equalized Odds: {:.2f}".format(phi_acc_robust_best,phi_robacc_robust_best, phi_eqodds_robust_best))
        phi_acc_robust_list.append(phi_acc_robust_best)
        phi_robacc_robust_list.append(phi_robacc_robust_best)
        phi_eqodds_robust_list.append(phi_eqodds_robust_best)

        # train with fairness
        model_fair = inprocessing.train_gpfair(args,model_fair,trainloader_poison)
        phi_acc_fair = testacc.test(args, model_fair, testloader)
        phi_robacc_fair = poisoning.test(args, testloader_poison, model_fair)
        phi_eqodds_fair = eqodds.test(args, model_fair, testloader)
        print("Test Accuracy: {:.2f}; Attack Success Rate (ASR): {:.2f}; Equalized Odds: {:.2f}".format(phi_acc_fair,phi_robacc_fair,phi_eqodds_fair))
        phi_acc_fair_list.append(phi_acc_fair) 
        phi_robacc_fair_list.append(phi_robacc_fair) 
        phi_eqodds_fair_list.append(phi_eqodds_fair)

        # pruning fair model for robustness
        thresholds = np.arange(0.6, 1.5, 0.05)
        temp_list_teacc, temp_list_asr, temp_list_eqodds = [], [], []
        for u in thresholds:
            print("Threshold: ", round(u,2))
            model_fair_robust = postprocessing.EP(model_fair.state_dict(), u, trainloader_poison, args, args.num_classes)
            phi_acc_robfair = testacc.test(args, model_fair_robust, testloader)
            phi_robacc_robfair = poisoning.test(args, testloader_poison, model_fair_robust)
            phi_eqodds_robfair = eqodds.test(args, model_fair_robust, testloader)
            print("Test Accuracy: {:.2f}; Attack Success Rate (ASR): {:.2f}; Equalized Odds: {:.2f}".format(phi_acc_robfair,phi_robacc_robfair,phi_eqodds_robfair))
            temp_list_teacc.append(phi_acc_robfair)
            temp_list_asr.append(phi_robacc_robfair)
            temp_list_eqodds.append(phi_eqodds_robfair)
        index_best = np.argmin(temp_list_asr)
        phi_acc_robfair_best = temp_list_teacc[index_best]
        phi_robacc_robfair_best = temp_list_asr[index_best]
        phi_eqodds_robfair_best = temp_list_eqodds[index_best]
        print("[Best] Test Accuracy: {:.2f}; Best Attack Success Rate (ASR): {:.2f}; Best Equalized Odds: {:.2f}".format(phi_acc_robfair_best,phi_robacc_robfair_best, phi_eqodds_robfair_best))
        phi_acc_fairrobust_list.append(phi_acc_robfair_best)
        phi_robacc_fairrobust_list.append(phi_robacc_robfair_best)
        phi_eqodds_fairrobust_list.append(phi_eqodds_robfair_best)


    phi_acc_poison_list, phi_robacc_poison_list, phi_eqodds_poison_list = np.array(phi_acc_poison_list), np.array(phi_robacc_poison_list), np.array(phi_eqodds_poison_list)
    phi_acc_robust_list, phi_robacc_robust_list, phi_eqodds_robust_list = np.array(phi_acc_robust_list), np.array(phi_robacc_robust_list), np.array(phi_eqodds_robust_list)
    phi_acc_fair_list, phi_robacc_fair_list, phi_eqodds_fair_list = np.array(phi_acc_fair_list), np.array(phi_robacc_fair_list), np.array(phi_eqodds_fair_list)
    phi_acc_fairrobust_list, phi_robacc_fairrobust_list, phi_eqodds_fairrobust_list = np.array(phi_acc_fairrobust_list), np.array(phi_robacc_fairrobust_list), np.array(phi_eqodds_fairrobust_list)


    print("[Poison + NoDef] Test Accuracy: {:.2f} $\pm$ {:.2f}; Attack Success Rate (ASR): {:.2f} $\pm$ {:.2f}; Equalized Odds: {:.2f} $\pm$ {:.2f}".format(phi_acc_poison_list.mean(), phi_acc_poison_list.std(), phi_robacc_poison_list.mean(), phi_robacc_poison_list.std(), phi_eqodds_poison_list.mean(), phi_eqodds_poison_list.std()))
    print("[Robust] Test Accuracy: {:.2f} $\pm$ {:.2f}; Attack Success Rate (ASR): {:.2f} $\pm$ {:.2f}; Equalized Odds: {:.2f} $\pm$ {:.2f}".format(phi_acc_robust_list.mean(), phi_acc_robust_list.std(), phi_robacc_robust_list.mean(), phi_robacc_robust_list.std(), phi_eqodds_robust_list.mean(), phi_eqodds_robust_list.std()))
    print("[Fair + Poison] Test Accuracy: {:.2f} $\pm$ {:.2f}; Attack Success Rate (ASR): {:.2f} $\pm$ {:.2f}; Equalized Odds: {:.2f} $\pm$ {:.2f}".format(phi_acc_fair_list.mean(), phi_acc_fair_list.std(), phi_robacc_fair_list.mean(), phi_robacc_fair_list.std(), phi_eqodds_fair_list.mean(), phi_eqodds_fair_list.std()))
    print("[Fair + Robust] Test Accuracy: {:.2f} $\pm$ {:.2f}; Attack Success Rate (ASR): {:.2f} $\pm$ {:.2f}; Equalized Odds: {:.2f} $\pm$ {:.2f}".format(phi_acc_fairrobust_list.mean(), phi_acc_fairrobust_list.std(), phi_robacc_fairrobust_list.mean(), phi_robacc_fairrobust_list.std(), phi_eqodds_fairrobust_list.mean(), phi_eqodds_fairrobust_list.std()))



def handle_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",type=str,default="UTKFACE",help="[UTKFACE]")
    parser.add_argument("--device",type=str,default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),help="GPU ID for this process")
    parser.add_argument("--epochs",type=int,default=30,help="number of epochs to train")
    parser.add_argument("--runs",type=int,default=5,help="number of iterations")

    # poisoning
    parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--trigger_label', type=int, default=1, help='The No. of trigger label (int, range from 0 to 10, default: 0)')
    parser.add_argument('--trigger_path', default="./defenses/outlier_robustness/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
    parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')
    parser.add_argument('--u', default=1.2, type=float,help='threshold hyperparameter')

    # group fairness
    parser.add_argument("--lamda", help = "Regularization hyperparameter", type = float, default = 1.0)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = handle_args()
    main(args)