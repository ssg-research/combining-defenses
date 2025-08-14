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

python -m standalone.evasion_inprocessing --dataset CENSUS > ./results/standalone/tabular/evasion_inprocessing_census.txt
# [No Defense] Test Accuracy: 80.28 $\pm$ 0.45; Robust Accuracy: 79.36 $\pm$ 0.33
# [Defense] Test Accuracy: 78.15 $\pm$ 0.00; Robust Accuracy: 77.43 $\pm$ 0.25

python -m standalone.gpfair_inprocessing --dataset CENSUS > ./results/standalone/gpfair_inprocessing_census.txt
# [No Defense] Test Accuracy: 79.77 $\pm$ 0.31; Equalized Odds: 1.28 $\pm$ 0.80
# [Defense] Test Accuracy: 75.84 $\pm$ 0.00; Equalized Odds: 0.00 $\pm$ 0.00

python -m standalone.expl_postprocessing --dataset CENSUS > ./results/standalone/expl_postprocessing_census.txt
# Test Accuracy: 80.08 $\pm$ 0.42; Error: 0.19 $\pm$ 0.05

python -m pairwise.c12_evasionRobIn_expl --dataset CENSUS > results/pairwise/c12_evasionRobIn_expl_census.txt
# [No Defense] Test Accuracy: 80.08 $\pm$ 0.49; Robust Accurcay: 80.04 $\pm$ 0.46; Error: 0.2092 $\pm$ 0.0304
# [Defense] Test Accuracy: 80.49 $\pm$ 1.32; Robust Accuracy: 80.19 $\pm$ 0.97; Error: 0.2092 $\pm$ 0.0305

python -m pairwise.c18_gpfairIn_expl --dataset CENSUS > results/pairwise/c18_gpfairIn_expl_census.txt
# [No Defense] Test Accuracy: 80.14 $\pm$ 0.47; Equalized Odds: 1.61 $\pm$ 0.60; Error: 0.1385 $\pm$ 0.0856
# [Defense] Test Accuracy: 77.51 $\pm$ 1.76; Equalized Odds: 1.21 $\pm$ 1.22; Error: 0.0077 $\pm$ 0.0030