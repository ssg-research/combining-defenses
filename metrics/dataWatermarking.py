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
import random
from PIL import Image
import numpy as np
from scipy.stats import ttest_rel
import torchvision.transforms as transforms

def get_preds(testloader, model, args):
    model.eval()
    for batch_idx, tuple_data in enumerate(testloader):
        inputs, targets = tuple_data[0].to(args.device), tuple_data[1].to(args.device)
        p = model(inputs)
        if args.dataset == "FMNIST":
            p = torch.nn.functional.softmax(p)
    return p

def train_with_watermark(args, model, testset_watermarked_new, testset_standard_new):

    Stats = [-1]*args.num_test
    p_value = [-1]*args.num_test

    for iters in range(args.num_test):
        # Random seed
        random.seed(random.randint(1, 10000))
        watermarked_loader = torch.utils.data.DataLoader(testset_watermarked_new, batch_size=100,shuffle=False, num_workers=1)
        standard_loader = torch.utils.data.DataLoader(testset_standard_new, batch_size=100,shuffle=False, num_workers=1)

        output_watermarked = get_preds(watermarked_loader, model, args)
        output_standard = get_preds(standard_loader, model, args)

        # export the target label
        target_select_water = [(output_watermarked[i, args.trigger_label]).cpu().detach().numpy() for i in range(len(output_watermarked))]
        target_select_stand = [(output_standard[i, args.trigger_label]).cpu().detach().numpy() for i in range(len(output_standard))]

        target_select_water = np.array(target_select_water)
        target_select_stand = np.array(target_select_stand)

        T_test = ttest_rel(target_select_stand + args.margin, target_select_water)

        Stats[iters], p_value[iters] = T_test[0], T_test[1]

        print("%i/%i" % (iters, args.num_test))
    idx_success_detection = [i for i in range(args.num_test) if (Stats[i] < 0) and (p_value[i] < 0.05 / 2)]  # single-sided hypothesis test
    rsd = float(len(idx_success_detection)) / args.num_test
    return rsd*100