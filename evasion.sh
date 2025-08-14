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

mkdir -p evasion/results

python -m evasion.standalone --dataset FMNIST --attack_class DLS --runs 3 > evasion/results/fmnist_DLS.txt
python -m evasion.standalone --dataset UTKFACE --runs 3 > evasion/results/utkface_autoattack.txt

python -m evasion.c9 --dataset FMNIST --attack_class DLS --runs 3 > evasion/results/c9_fmnist_DLS.txt
python -m evasion.c9 --dataset UTKFACE --runs 3 > evasion/results/c9_utkface_autoattack.txt

python -m evasion.c12 --dataset FMNIST --attack_class DLS --runs 3 > evasion/results/c12_fmnist_DLS.txt
python -m evasion.c12 --dataset UTKFACE --runs 3 > evasion/results/c12_utkface_autoattack.txt

python -m evasion.c35 --dataset FMNIST --attack_class DLS --runs 3 > evasion/results/c33_fmnist_DLS.txt
python -m evasion.c35 --dataset UTKFACE --runs 3 > evasion/results/c33_utkface_autoattack.txt