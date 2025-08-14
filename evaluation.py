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

from sklearn.metrics import balanced_accuracy_score, accuracy_score

ground_truth_prior = [1, 1, 1, 0, 0, 0, 1, 1]
naive_approach_prior = [0, 1, 1, 1, 1, 1, 1, 1]
our_approach_prior = [1, 1, 1, 0, 0, 0, 0, 1]

print("########### Prior Work Ground Truth #############")
print("[Accuracy] Naive Accuracy", accuracy_score(ground_truth_prior,naive_approach_prior))
print("[Accuracy] Our Accuracy", accuracy_score(ground_truth_prior,our_approach_prior))

print("[Balanced Accuracy] Naive Accuracy", balanced_accuracy_score(ground_truth_prior,naive_approach_prior))
print("[Balanced Accuracy] Our Accuracy", balanced_accuracy_score(ground_truth_prior,our_approach_prior))



ground_truth_empirical = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
naive_approach_empirical = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1]
our_approach_empirical = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]

print("########### Empirical Ground Truth #############")
print("[Accuracy] Naive Accuracy", accuracy_score(ground_truth_empirical,naive_approach_empirical))
print("[Accuracy] Our Accuracy", accuracy_score(ground_truth_empirical,our_approach_empirical))

print("[Balanced Accuracy] Naive Accuracy", balanced_accuracy_score(ground_truth_empirical,naive_approach_empirical))
print("[Balanced Accuracy] Our Accuracy", balanced_accuracy_score(ground_truth_empirical,our_approach_empirical))

ground_truth_empirical = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

print("########### Empirical Ground Truth (After hyperparameter tuning) #############")
print("[Accuracy] Naive Accuracy", accuracy_score(ground_truth_empirical,naive_approach_empirical))
print("[Accuracy] Our Accuracy", accuracy_score(ground_truth_empirical,our_approach_empirical))

print("[Balanced Accuracy] Naive Accuracy", balanced_accuracy_score(ground_truth_empirical,naive_approach_empirical))
print("[Balanced Accuracy] Our Accuracy", balanced_accuracy_score(ground_truth_empirical,our_approach_empirical))