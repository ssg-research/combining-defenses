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

from defenses.outlier_robustness import utils_outrem, inprocessing
from defenses.fingerprint import postprocessing

from metrics import testacc, poisoning
import models, utils, data


def main(args):
    
    dataload = {"FMNIST": data.process_fmnist,"UTKFACE": data.process_utkface}
    if args.dataset == "FMNIST":
        traindata_clean, testdata_clean = dataload[args.dataset]()
        trainset_poison, testset_poisoned = utils_outrem.generate_dataset(args, traindata_clean, testdata_clean)
        indices = list(range(0, len(traindata_clean), 10))
        val_data = torch.utils.data.Subset(traindata_clean, indices)
    else:
        traindata_clean, testdata_clean, val_data, X_train, y_train, X_val, y_val, X_test, y_test, Z_train, Z_test, Z_val = dataload[args.dataset](lists=True, valdata=True)
        traindata = (torch.from_numpy(np.array(X_train)).type(torch.FloatTensor), torch.from_numpy(np.array(y_train)).type(torch.LongTensor), torch.from_numpy(np.array(Z_train)).type(torch.FloatTensor))
        testdata = (torch.from_numpy(np.array(X_test)).type(torch.FloatTensor), torch.from_numpy(np.array(y_test)).type(torch.LongTensor), torch.from_numpy(np.array(Z_test)).type(torch.FloatTensor))
        trainset_poison, testset_poisoned = utils_outrem.generate_dataset(args, traindata, testdata)

    trainloader = torch.utils.data.DataLoader(dataset=traindata_clean, batch_size=256, shuffle=True)
    valloader = torch.utils.data.DataLoader(dataset=val_data, batch_size=256, shuffle=True)
    trainloader_poison = torch.utils.data.DataLoader(dataset=trainset_poison, batch_size=256, shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset=testdata_clean, batch_size=256, shuffle=True)
    testloader_poison = torch.utils.data.DataLoader(dataset=testset_poisoned, batch_size=256, shuffle=True)

    phi_acc_poison_list, phi_robacc_poison_list = [], []
    phi_acc_def_list, phi_robacc_def_list = [], []
    phi_pval_nodef_list,phi_pval_def_list = [], []
    for i in range(args.runs):
        if args.dataset == "FMNIST":
            args.epochs = 15
            model_nodef_poison = models.FashionMNIST().to(args.device)
            model_robust = models.FashionMNIST().to(args.device)
            args.num_classes=10
            dataset_dim = '2D'
        else:
            model_nodef_poison = models.VGGBinary('VGG16').to(args.device)
            model_robust = models.VGGBinary('VGG16').to(args.device)
            args.num_classes=2
            args.epochs = 10
            dataset_dim = '2D'

        model_nodef_poison = utils_outrem.train_backdoor(args,model_nodef_poison, trainloader_poison, testloader_poison)
        phi_acc_poison = testacc.test(args, model_nodef_poison, testloader)
        phi_robacc_poison = poisoning.test(args, testloader_poison, model_nodef_poison)
        fingerprinting = postprocessing.Fingerprinting(args,
                                    model_nodef_poison,
                                    trainloader_poison,
                                    testloader,
                                    args.num_classes,
                                    args.device,
                                    args.distance,
                                    dataset_dim,
                                    args.alpha_l1,
                                    args.alpha_l2,
                                    args.alpha_linf,
                                    args.k,
                                    args.gap,
                                    args.num_iter,
                                    args.regressor_embed,
                                    args.val_size,
                                    256
                                    )
        results = fingerprinting.dataset_inference()
        pval_nodef = results["target"]['p-value'] 
        phi_pval_nodef_list.append(pval_nodef)
        print("Test Accuracy: {:.2f}; Attack Success Rate (ASR): {:.2f}; pval: {}".format(phi_acc_poison,phi_robacc_poison, pval_nodef))
        phi_acc_poison_list.append(phi_acc_poison) 
        phi_robacc_poison_list.append(phi_robacc_poison)

        model_robust = inprocessing.finetune_setup(args, model_nodef_poison)
        args.epochs = 5
        model_robust = utils.train(args, model_robust, valloader)
        phi_acc_def = testacc.test(args, model_robust, testloader)
        phi_robacc_def = poisoning.test(args, testloader_poison, model_robust)
        fingerprinting = postprocessing.Fingerprinting(args,
                                    model_robust,
                                    trainloader_poison,
                                    testloader,
                                    args.num_classes,
                                    args.device,
                                    args.distance,
                                    dataset_dim,
                                    args.alpha_l1,
                                    args.alpha_l2,
                                    args.alpha_linf,
                                    args.k,
                                    args.gap,
                                    args.num_iter,
                                    args.regressor_embed,
                                    args.val_size,
                                    256
                                    )
        results = fingerprinting.dataset_inference()
        pval_def = results["target"]['p-value'] 
        phi_pval_def_list.append(pval_def)
        print("Test Accuracy: {:.2f}; Attack Success Rate (ASR): {:.2f}; pval: {}".format(phi_acc_def,phi_robacc_def,pval_def))
        phi_acc_def_list.append(phi_acc_def) 
        phi_robacc_def_list.append(phi_robacc_def)

    phi_acc_poison_list, phi_robacc_poison_list = np.array(phi_acc_poison_list), np.array(phi_robacc_poison_list)
    phi_acc_def_list, phi_robacc_def_list = np.array(phi_acc_def_list), np.array(phi_robacc_def_list)
    phi_pval_nodef_list,phi_pval_def_list = np.array(phi_pval_nodef_list), np.array(phi_pval_def_list)

    print("[No Defense] Test Accuracy: {:.2f} $\pm$ {:.2f}; Attack Success Rate (ASR): {:.2f} $\pm$ {:.2f} pval: {} $\pm$ {}".format(phi_acc_poison_list.mean(), phi_acc_poison_list.std(), phi_robacc_poison_list.mean(), phi_robacc_poison_list.std(), phi_pval_nodef_list.mean(), phi_pval_nodef_list.std()))
    print("[Defense] Test Accuracy: {:.2f} $\pm$ {:.2f}; Attack Success Rate (ASR): {:.2f} $\pm$ {:.2f} pval: {} $\pm$ {}".format(phi_acc_def_list.mean(), phi_acc_def_list.std(), phi_robacc_def_list.mean(), phi_robacc_def_list.std(), phi_pval_def_list.mean(), phi_pval_def_list.std()))



def handle_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",type=str,default="FMNIST",help="[FMNIST,UTKFACE]")
    parser.add_argument("--device",type=str,default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),help="GPU ID for this process")
    parser.add_argument("--epochs",type=int,default=5,help="number of epochs to train")
    parser.add_argument("--runs",type=int,default=5,help="number of iterations")

    # poisoning
    parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
    parser.add_argument('--trigger_path', default="./defenses/outlier_robustness/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
    parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')

    ##### DI parameters
    parser.add_argument('--num_iter', type=int, default=50)
    parser.add_argument("--distance", help="Type of Adversarial Perturbation", type=str)#, choices = ["linf", "l1", "l2", "vanilla"])
    parser.add_argument("--randomize", help = "For the individual attacks", type = int, default = 0, choices = [0,1,2])
    parser.add_argument("--alpha_l1", help = "Step Size for L1 attacks", type = float, default = 1.0)
    parser.add_argument("--alpha_l2", help = "Step Size for L2 attacks", type = float, default = 0.01)
    parser.add_argument("--alpha_linf", help = "Step Size for Linf attacks", type = float, default = 0.001)
    parser.add_argument("--gap", help = "For L1 attack", type = float, default = 0.001)
    parser.add_argument("--k", help = "For L1 attack", type = int, default = 1)
    parser.add_argument("--val_size", help = "Validation Size", type = int, default = 100)
    parser.add_argument("--regressor_embed", help = "Victim Embeddings for training regressor", type = int, default = 0, choices = [0,1])

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = handle_args()
    main(args)