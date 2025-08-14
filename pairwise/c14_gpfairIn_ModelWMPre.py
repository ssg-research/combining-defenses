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

# Model watermarking (preprocessing) with group fairness (in-processing)

import torch
import argparse
import numpy as np

from defenses.model_watermarking import preprocessing
from defenses.group_fairness import inprocessing
from metrics import testacc, watermarking, eqodds
import models, utils, data


def main(args):
    
    dataload = {"UTKFACE": data.process_utkface}

    traindata_clean, testdata_clean, X_train, y_train, X_test, y_test, Z_train, Z_test = dataload[args.dataset](lists=True)
    traindata = (torch.from_numpy(np.array(X_train)).type(torch.FloatTensor), torch.from_numpy(np.array(y_train)).type(torch.LongTensor), torch.from_numpy(np.array(Z_train)).type(torch.FloatTensor))
    testdata = (torch.from_numpy(np.array(X_test)).type(torch.FloatTensor), torch.from_numpy(np.array(y_test)).type(torch.LongTensor), torch.from_numpy(np.array(Z_test)).type(torch.FloatTensor))
    trainset_poison, testset_poisoned = preprocessing.generate_watermarked_dataset(args, traindata, testdata)

    trainloader = torch.utils.data.DataLoader(dataset=traindata_clean, batch_size=256, shuffle=True)
    trainloader_poison = torch.utils.data.DataLoader(dataset=trainset_poison, batch_size=256, shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset=testdata_clean, batch_size=256, shuffle=True)
    testloader_poison = torch.utils.data.DataLoader(dataset=testset_poisoned, batch_size=256, shuffle=True)


    phi_acc_nodef_list, phi_wmacc_nodef_list, phi_eqodds_nodef_list = [], [], []
    phi_acc_wm_list, phi_wmacc_wm_list, phi_eqodds_wm_list = [], [], []
    phi_acc_fair_list, phi_wmacc_fair_list, phi_eqodds_fair_list = [], [], []
    phi_acc_wmfair_list, phi_wmacc_wmfair_list, phi_eqodds_wmfair_list = [], [], []
    for i in range(args.runs):
        model_nodef = models.VGGBinary('VGG16').to(args.device)
        model_wm = models.VGGBinary('VGG16').to(args.device)
        model_fair = models.VGGBinary('VGG16').to(args.device)
        model_wm_fair = models.VGGBinary('VGG16').to(args.device)
        args.num_classes=2

        # No defense baseline
        model_nodef = utils.train(args, model_nodef, trainloader)
        phi_acc_nodef = testacc.test(args, model_nodef, testloader)
        phi_wmacc_nodef = watermarking.test_backdoor(args, testloader_poison, model_nodef)
        phi_eqodds_nodef = eqodds.test(args, model_nodef, testloader)
        print("Test Accuracy: {:.2f}; Watermark Accuracy: {:.2f}; Equalized Odds: {:.2f}".format(phi_acc_nodef,phi_wmacc_nodef, phi_eqodds_nodef))
        phi_acc_nodef_list.append(phi_acc_nodef)
        phi_wmacc_nodef_list.append(phi_wmacc_nodef)
        phi_eqodds_nodef_list.append(phi_eqodds_nodef)

        # Only watermarking baseline
        model_wm = preprocessing.train_watermark(args,model_wm, trainloader_poison, testloader_poison)
        phi_acc_wm = testacc.test(args, model_wm, testloader)
        phi_wmacc_wm = watermarking.test_backdoor(args, testloader_poison, model_wm)
        phi_eqodds_wm = eqodds.test(args, model_fair, testloader)
        print("Test Accuracy: {:.2f}; Watermark Accuracy: {:.2f}; Equalized Odds: {:.2f}".format(phi_acc_wm,phi_wmacc_wm, phi_eqodds_wm))
        phi_acc_wm_list.append(phi_acc_wm)
        phi_wmacc_wm_list.append(phi_wmacc_wm) 
        phi_eqodds_wm_list.append(phi_eqodds_wm)

        # Only fairness baseline
        model_fair = inprocessing.train_gpfair(args,model_fair,trainloader)
        phi_acc_fair = testacc.test(args, model_fair, testloader)
        phi_wmacc_fair = watermarking.test_backdoor(args, testloader_poison, model_fair)
        phi_eqodds_fair = eqodds.test(args, model_fair, testloader)
        print("Test Accuracy: {:.2f}; Watermark Accuracy: {:.2f}; Equalized Odds: {:.2f}".format(phi_acc_fair,phi_wmacc_fair,phi_eqodds_fair))
        phi_acc_fair_list.append(phi_acc_fair)
        phi_wmacc_fair_list.append(phi_wmacc_fair)
        phi_eqodds_fair_list.append(phi_eqodds_fair)


        # both watermarking and group fairness
        model_wm_fair = inprocessing.train_gpfair(args,model_wm_fair,trainloader_poison)
        phi_acc_wmfair = testacc.test(args, model_wm_fair, testloader)
        phi_wmacc_wmfair = watermarking.test_backdoor(args, testloader_poison, model_wm_fair)
        phi_eqodds_wmfair = eqodds.test(args, model_wm_fair, testloader)
        print("Test Accuracy: {:.2f}; Watermark Accuracy: {:.2f}; Equalized Odds: {:.2f}".format(phi_acc_wmfair, phi_wmacc_wmfair, phi_eqodds_wmfair))
        phi_acc_wmfair_list.append(phi_acc_wmfair)
        phi_wmacc_wmfair_list.append(phi_wmacc_wmfair)
        phi_eqodds_wmfair_list.append(phi_eqodds_wmfair)


    phi_acc_nodef_list, phi_wmacc_nodef_list, phi_eqodds_nodef_list = np.array(phi_acc_nodef_list), np.array(phi_wmacc_nodef_list), np.array(phi_eqodds_nodef_list)
    phi_acc_wm_list, phi_wmacc_wm_list, phi_eqodds_wm_list = np.array(phi_acc_wm_list), np.array(phi_wmacc_wm_list), np.array(phi_eqodds_wm_list)
    phi_acc_fair_list, phi_wmacc_fair_list, phi_eqodds_fair_list = np.array(phi_acc_fair_list), np.array(phi_wmacc_fair_list), np.array(phi_eqodds_fair_list)
    phi_acc_wmfair_list, phi_wmacc_wmfair_list, phi_eqodds_wmfair_list = np.array(phi_acc_wmfair_list), np.array(phi_wmacc_wmfair_list), np.array(phi_eqodds_wmfair_list)

    print("[No Defense] Test Accuracy: {:.2f} $\pm$ {:.2f}; Watermarking Accurcay: {:.2f} $\pm$ {:.2f}; Equalized Odds: {:.2f} $\pm$ {:.2f}".format(phi_acc_nodef_list.mean(), phi_acc_nodef_list.std(), phi_wmacc_nodef_list.mean(), phi_wmacc_nodef_list.std(), phi_eqodds_nodef_list.mean(), phi_eqodds_nodef_list.std()))
    print("[Watermarking] Test Accuracy: {:.2f} $\pm$ {:.2f}; Watermarking Accuracy: {:.2f} $\pm$ {:.2f}; Equalized Odds: {:.2f} $\pm$ {:.2f}".format(phi_acc_wm_list.mean(), phi_acc_wm_list.std(), phi_wmacc_wm_list.mean(), phi_wmacc_wm_list.std(), phi_eqodds_wm_list.mean(), phi_eqodds_wm_list.std()))
    print("[Fair] Test Accuracy: {:.2f} $\pm$ {:.2f}; Watermarking Accurcay: {:.2f} $\pm$ {:.2f}; Equalized Odds: {:.2f} $\pm$ {:.2f}".format(phi_acc_fair_list.mean(), phi_acc_fair_list.std(), phi_wmacc_fair_list.mean(), phi_wmacc_fair_list.std(), phi_eqodds_fair_list.mean(), phi_eqodds_fair_list.std()))
    print("[Fair + Watermarking] Test Accuracy: {:.2f} $\pm$ {:.2f}; Watermarking Accuracy: {:.2f} $\pm$ {:.2f}; Equalized Odds: {:.2f} $\pm$ {:.2f}".format(phi_acc_wmfair_list.mean(), phi_acc_wmfair_list.std(), phi_wmacc_wmfair_list.mean(), phi_wmacc_wmfair_list.std(), phi_eqodds_wmfair_list.mean(), phi_eqodds_wmfair_list.std()))


def handle_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",type=str,default="UTKFACE",help="[UTKFACE]")
    parser.add_argument("--device",type=str,default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),help="GPU ID for this process")
    parser.add_argument("--epochs",type=int,default=5,help="number of epochs to train")
    parser.add_argument("--runs",type=int,default=5,help="number of iterations")

    # poisoning
    parser.add_argument('--wm_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
    parser.add_argument('--trigger_path', default="./defenses/outlier_robustness/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
    parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')

    # group fairness
    parser.add_argument("--lamda", help = "Regularization hyperparameter", type = float, default = 1.0)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = handle_args()
    main(args)
