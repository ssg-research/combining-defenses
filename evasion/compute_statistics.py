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

import pandas as pd

# Compute metrics
def compute_metrics(df, heading):
    attack_success_rate = df["attack_success"].mean()
    robacc = 1-attack_success_rate
    num_unavailable_gradients = df["unavailable_gradients"].sum()
    num_unstable_predictions_gt_0_1 = (df["unstable_predictions"] > 0.1).sum()
    num_silent_success = df["silent_success"].sum()
    num_incomplete_optimization_true = df["incomplete_optimization"].astype(bool).sum()
    num_transfer_failure = df["transfer_failure"].sum()
    num_unconstrained_attack_failure_1_0 = (df["unconstrained_attack_failure"] == 1.0).sum()

    # Print results
    print("\n\n####### {} #######\n".format(heading))
    print(f"RA: {robacc:.2%}")
    print(f"I1: Unavailable Gradients (True): {num_unavailable_gradients}")
    print(f"I2: Unstable Predictions > 0.1: {num_unstable_predictions_gt_0_1}")
    print(f"I3: Silent Success (True): {num_silent_success}")
    print(f"I4: Incomplete Optimization (True): {num_incomplete_optimization_true}")
    print(f"I5: Transfer Failure (True): {num_transfer_failure}")
    print(f"I6: Unconstrained Attack Failure == 1.0: {num_unconstrained_attack_failure_1_0}")


df_nodef = pd.read_csv("./results/c12_indicators_FMNIST_DLS_rand_def.csv")
compute_metrics(df_nodef, "No Defense")
# df_def = pd.read_csv("./results/c35_indicators_FMNIST_DLS_rand_def.csv")
# compute_metrics(df_def, "With Defense")