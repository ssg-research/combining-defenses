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
from torch.utils.data import Subset

from defenses.model_watermarking import inprocessing
from metrics import testacc, watermarking
import models, utils, data


def main(args):
    
    dataload = {"FMNIST": data.process_fmnist,"UTKFACE": data.process_utkface}
    if args.dataset == "FMNIST":
        traindata, testdata = dataload[args.dataset]()
    else:
        _, _, X_train, y_train, X_test, y_test, _, _ = dataload[args.dataset](lists=True)
        traindata = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_train)).type(torch.FloatTensor), torch.from_numpy(np.array(y_train)).type(torch.LongTensor))
        testdata = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_test)).type(torch.FloatTensor), torch.from_numpy(np.array(y_test)).type(torch.LongTensor))

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=256, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=256, shuffle=False, num_workers=1)
    
    traindata_temp = Subset(traindata, list(range(len(traindata) // 2)))
    if args.dataset == "FMNIST":
        watermarkset = inprocessing.watermark_textoverlay(args, traindata_temp, count=int(args.wm_rate*len(traindata)))
    else:
        watermarkset = inprocessing.watermark_textoverlay_wsattr(args, traindata_temp, count=int(args.wm_rate*len(traindata)))
    wmloader = torch.utils.data.DataLoader(watermarkset, batch_size=256, shuffle=True, num_workers=1)

    def my_collate(batch):
        """Define collate_fn myself because the default_collate_fn throws errors like crazy"""
        # item: a tuple of (img, label)
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        data = torch.stack(data)
        target = torch.LongTensor(target)
        return [data, target]

    train_watermark_mixset = torch.utils.data.ConcatDataset((traindata_temp, watermarkset))
    train_watermark_loader = torch.utils.data.DataLoader(train_watermark_mixset, batch_size=256, collate_fn = my_collate, shuffle=True, num_workers=1)

    phi_acc_nodef_list,phi_wm_nodef_list = [], []
    phi_acc_def_list,phi_wm_def_list = [], []
    for i in range(args.runs):
        if args.dataset == "FMNIST":
            model_nodef = models.FashionMNIST().to(args.device)
            model_wm = models.FashionMNIST().to(args.device)
            args.epochs = 15
            args.num_classes=10
            args.warmup_epochs = 12
        else:
            model_nodef = models.VGGBinary('VGG16').to(args.device)
            model_wm = models.VGGBinary('VGG16').to(args.device)
            args.epochs = 5
            args.num_classes=2

        model_nodef = utils.train(args, model_nodef, trainloader)
        phi_acc_nodef = testacc.test(args, model_nodef, testloader)
        phi_wmacc_nodef = watermarking.test_inproc(args, model_nodef, wmloader)
        print("Test Accuracy: {:.2f}; Watermarking Accuracy: {:.2f}".format(phi_acc_nodef,phi_wmacc_nodef))
        phi_acc_nodef_list.append(phi_acc_nodef)
        phi_wm_nodef_list.append(phi_wmacc_nodef)

        model_wm = inprocessing.train_watermark(args, model_wm, train_watermark_loader, wmloader)
        phi_acc_wm = testacc.test(args, model_wm, testloader)
        phi_wmacc_wm = watermarking.test_backdoor(args, wmloader, model_wm)
        print("Test Accuracy: {:.2f}; Watermarking Accuracy: {:.2f}".format(phi_acc_wm,phi_wmacc_wm))
        phi_acc_def_list.append(phi_acc_wm)
        phi_wm_def_list.append(phi_wmacc_wm)

    phi_acc_nodef_list,phi_wm_nodef_list = np.array(phi_acc_nodef_list), np.array(phi_wm_nodef_list)
    phi_acc_def_list, phi_wm_def_list = np.array(phi_acc_def_list), np.array(phi_wm_def_list)

    print("[No Defense] Test Accuracy: {:.2f} $\pm$ {:.2f}; Watermarking Accuracy: {:.2f} $\pm$ {:.2f}".format(phi_acc_nodef_list.mean(), phi_acc_nodef_list.std(), phi_wm_nodef_list.mean(), phi_wm_nodef_list.std()))
    print("[Defense] Test Accuracy: {:.2f} $\pm$ {:.2f}; Watermarking Accuracy: {:.2f} $\pm$ {:.2f}".format(phi_acc_def_list.mean(), phi_acc_def_list.std(), phi_wm_def_list.mean(), phi_wm_def_list.std()))


def handle_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",type=str,default="FMNIST",help="[FMNIST,UTKFACE]")
    parser.add_argument("--device",type=str,default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),help="GPU ID for this process")
    parser.add_argument("--epochs",type=int,default=5,help="number of epochs to train")
    parser.add_argument("--runs",type=int,default=5,help="number of iterations")
    
    # poisoning
    parser.add_argument('--wm_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
    parser.add_argument("--warmup_epochs", default=5, type=int)  
    parser.add_argument("--robust_noise", default=1.0, type=float)
    parser.add_argument("--robust_noise_step", default=0.05, type=float)
    parser.add_argument('--simple', action="store_true")
    parser.add_argument("--avgtimes", default=100, type=int)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = handle_args()
    main(args)