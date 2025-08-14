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
import numpy as np

def equalized_odds(y_pred: np.ndarray, y_gt: np.ndarray, sensitive_attribute: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculates the equalized odds, which measures the difference in true positive rate and false positive rate
    between different values of a binary sensitive attribute.

    Args:
        y_pred: A 1D array of predicted probabilities between 0 and 1 for each data point.
        y_gt: A 1D array of binary ground truth labels (0 or 1) for each data point.
        sensitive_attribute: A 1D array of binary values (0 or 1) indicating the sensitive attribute for each data point.
        threshold: A float threshold value for converting predicted probabilities to binary predictions.

    Returns:
        A float value between 0 and 100 representing the percentage difference in true positive rate and false positive rate
        between different values of the sensitive attribute.
    """
    # Make a copy of the predicted probabilities and sensitive attributes.
    y_pred_all = y_pred.copy()
    sensitive_attribute_all = sensitive_attribute.copy()

    # Select the predicted probabilities and sensitive attributes for the data points where the ground truth is positive.
    y_pred = y_pred_all[y_gt == 1]
    sensitive_attribute = sensitive_attribute_all[y_gt == 1]

    # Convert predicted probabilities to binary predictions using the threshold and calculate the difference in true positive rate.
    y_z_1 = y_pred[sensitive_attribute == 1] > threshold if threshold else y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0] > threshold if threshold else y_pred[sensitive_attribute == 0]
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0
    equality = abs(y_z_1.mean() - y_z_0.mean())

    # Select the predicted probabilities and sensitive attributes for the data points where the ground truth is negative.
    y_pred = y_pred_all[y_gt == 0]
    sensitive_attribute = sensitive_attribute_all[y_gt == 0]

    # Convert predicted probabilities to binary predictions using the threshold and calculate the difference in false positive rate.
    y_z_1 = y_pred[sensitive_attribute == 1] > threshold if threshold else y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0] > threshold if threshold else y_pred[sensitive_attribute == 0]
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0
    equality += abs(y_z_1.mean() - y_z_0.mean())

    # Multiply the difference in true positive rate and false positive rate by 100 to get a percentage difference and return it.
    equality *= 100
    return equality

def test(args, model, test_loader):

    with torch.no_grad():
        model.eval()
        target_hat_list = []
        target_list = []
        sensitive_list = []

        for (X, target, sensitive) in test_loader:
            
            data = X.to(args.device)
            target = target.to(args.device).unsqueeze(1).type(torch.long)
            sensitive = sensitive.to(args.device).unsqueeze(1).type(torch.float)

            output = model(data)
            _, pred = torch.max(output,1)
            pred = pred.unsqueeze(1)

            target_hat_list.append(pred.cpu().numpy())
            target_list.append(target.cpu().numpy())
            sensitive_list.append(sensitive.cpu().numpy())

        target_hat_list = np.concatenate(target_hat_list, axis=0)
        target_list = np.concatenate(target_list, axis=0)
        sensitive_list = np.concatenate(sensitive_list, axis=0)

    eqodds = equalized_odds(y_gt=target_list, y_pred=target_hat_list, sensitive_attribute=sensitive_list)

    return eqodds