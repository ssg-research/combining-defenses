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

from defenses.model_watermarking import inprocessing, preprocessing
from metrics import testacc, watermarking, dataWatermarking
import models, utils, data


def main(args):
    
    dataload = {"FMNIST": data.process_fmnist,"UTKFACE": data.process_utkface}
    if args.dataset == "FMNIST":
        traindata_clean, testdata_clean = dataload[args.dataset]()
        trainset_wmData, testset_wmData = preprocessing.generate_watermarked_dataset(args, traindata_clean, testdata_clean)
    else:
        traindata_clean, testdata_clean, X_train, y_train, X_test, y_test, Z_train, Z_test = dataload[args.dataset](lists=True)
        traindata_clean = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_train)).type(torch.FloatTensor), torch.from_numpy(np.array(y_train)).type(torch.LongTensor))
        testdata_clean = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_test)).type(torch.FloatTensor), torch.from_numpy(np.array(y_test)).type(torch.LongTensor))
        traindata = (torch.from_numpy(np.array(X_train)).type(torch.FloatTensor), torch.from_numpy(np.array(y_train)).type(torch.LongTensor))
        testdata = (torch.from_numpy(np.array(X_test)).type(torch.FloatTensor), torch.from_numpy(np.array(y_test)).type(torch.LongTensor))
        trainset_wmData = preprocessing.UTKWatermarkNoSattr(args = args, dataset = traindata, train=True)
        testset_wmData = preprocessing.UTKWatermarkNoSattr(args=args, dataset=testdata, train=False)

    trainloader = torch.utils.data.DataLoader(traindata_clean, batch_size=256, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testdata_clean, batch_size=256, shuffle=False, num_workers=1)
    
    traindata_temp = Subset(trainset_wmData, list(range(len(trainset_wmData) // 2)))
    if args.dataset == "FMNIST":
        watermarkset = inprocessing.watermark_textoverlay(args, traindata_temp, count=int(args.wm_rate*len(trainset_wmData)))
    else:
        X_wm,y_wm = inprocessing.watermark_textoverlay_wsattr_new(args, traindata_temp, count=int(args.wm_rate*len(trainset_wmData)))
        X_wm = np.array([x.numpy() for x in X_wm]) 
        y_wm = np.array(y_wm)
        watermarkset = torch.utils.data.TensorDataset(torch.from_numpy(X_wm).type(torch.FloatTensor), torch.from_numpy(y_wm).type(torch.LongTensor))

    wmloader = torch.utils.data.DataLoader(watermarkset, batch_size=256, shuffle=True, num_workers=1)

    def my_collate(batch):
        """Define collate_fn myself because the default_collate_fn throws errors like crazy"""
        # item: a tuple of (img, label)
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        data = torch.stack(data)
        target = torch.LongTensor(target)
        return [data, target]

    train_watermark_mixset = torch.utils.data.ConcatDataset((trainset_wmData, watermarkset))
    train_watermark_loader = torch.utils.data.DataLoader(train_watermark_mixset, batch_size=256, collate_fn = my_collate, shuffle=True, num_workers=1)

    phi_acc_def_list,phi_wmacc_def_list,phi_rsd_def_list = [], [], []
    for i in range(args.runs):
        if args.dataset == "FMNIST":
            model_wm = models.FashionMNIST().to(args.device)
            args.epochs = 15
            args.num_classes=10
            args.warmup_epochs = 12
        else:
            model_wm = models.VGGBinary('VGG16').to(args.device)
            args.epochs = 10
            args.warmup_epochs = 10
            args.num_classes=2
            args.margin = 0



        if args.dataset == "FMNIST":
            model_wm = inprocessing.train_watermark(args, model_wm, train_watermark_loader, wmloader)
        else:
            model_wm = utils.train(args, model_wm, train_watermark_loader)
        phi_acc_def = testacc.test(args, model_wm, testloader)
        phi_wmacc_def = watermarking.test_backdoor(args, wmloader, model_wm)
        phi_rsd_def = dataWatermarking.train_with_watermark(args, model_wm, testset_wmData, testdata_clean)
        print("Test Accuracy: {:.2f}; Watermark Accuracy: {:.2f}; RSD: {:.2f}".format(phi_acc_def,phi_wmacc_def,phi_rsd_def))
        phi_acc_def_list.append(phi_acc_def)
        phi_wmacc_def_list.append(phi_wmacc_def)
        phi_rsd_def_list.append(phi_rsd_def)
        
    phi_acc_def_list, phi_wmacc_def_list, phi_rsd_def_list = np.array(phi_acc_def_list), np.array(phi_wmacc_def_list), np.array(phi_rsd_def_list)
    print("[Defense] Test Accuracy: {:.2f} $\pm$ {:.2f}; WMAcc: {:.2f} $\pm$ {:.2f}; RSD: {:.2f} $\pm$ {:.2f}".format(phi_acc_def_list.mean(), phi_acc_def_list.std(), phi_wmacc_def_list.mean(), phi_wmacc_def_list.std(), phi_rsd_def_list.mean(), phi_rsd_def_list.std()))



def handle_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",type=str,default="FMNIST",help="[FMNIST,UTKFACE]")
    parser.add_argument("--device",type=str,default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),help="GPU ID for this process")
    parser.add_argument("--epochs",type=int,default=5,help="number of epochs to train")
    parser.add_argument("--runs",type=int,default=5,help="number of runs")

    # poisoning
    parser.add_argument('--wm_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
    parser.add_argument("--warmup_epochs", default=5, type=int)  
    parser.add_argument("--robust_noise", default=1.0, type=float)
    parser.add_argument("--robust_noise_step", default=0.05, type=float)
    parser.add_argument('--simple', action="store_true")
    parser.add_argument("--avgtimes", default=100, type=int)

    parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
    parser.add_argument('--trigger_path', default="./defenses/outlier_robustness/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
    parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')
    parser.add_argument('--num_test', default=100, type=int,help='number of T-test')
    parser.add_argument('--margin', default=0.2, type=float, help='the margin in the pairwise T-test')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = handle_args()
    main(args)