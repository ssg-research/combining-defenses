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

python -m evasion.standalone_indicators --dataset FMNIST --attack_class DLS > evasion/results/fmnist_DLS.txt
python -m evasion.standalone_indicators --dataset FMNIST --attack_class CE > evasion/results/fmnist_CE.txt
python -m evasion.standalone_indicators --dataset FMNIST --attack_class CE --version rand > evasion/results/fmnist_CE_rand.txt
python -m evasion.standalone_indicators --dataset FMNIST --attack_class DLS --version rand > evasion/results/fmnist_DLS_rand.txt


python -m evasion.c9_indicators --dataset FMNIST --attack_class DLS > evasion/results/c9_fmnist_DLS.txt
python -m evasion.c9_indicators --dataset FMNIST --attack_class CE > evasion/results/c9_fmnist_CE.txt
python -m evasion.c9_indicators --dataset FMNIST --attack_class DLS --version rand > evasion/results/c9_fmnist_DLS_rand.txt
python -m evasion.c9_indicators --dataset FMNIST --attack_class CE --version rand > evasion/results/c9_fmnist_CE_rand.txt


python -m evasion.c12_indicators --dataset FMNIST --attack_class DLS > evasion/results/c12_fmnist_DLS.txt
python -m evasion.c12_indicators --dataset FMNIST --attack_class CE > evasion/results/c12_fmnist_CE.txt
python -m evasion.c12_indicators --dataset FMNIST --attack_class DLS --version rand > evasion/results/c12_fmnist_DLS_rand.txt
python -m evasion.c12_indicators --dataset FMNIST --attack_class CE --version rand > evasion/results/c12_fmnist_CE_rand.txt


python -m evasion.c35_indicators --dataset FMNIST --attack_class DLS > evasion/results/c35_fmnist_DLS.txt
python -m evasion.c35_indicators --dataset FMNIST --attack_class CE > evasion/results/c35_fmnist_CE.txt
python -m evasion.c35_indicators --dataset FMNIST --attack_class DLS --version rand > evasion/results/c35_fmnist_DLS_rand.txt
python -m evasion.c35_indicators --dataset FMNIST --attack_class CE --version rand > evasion/results/c35_fmnist_CE_rand.txt

#not used
# python -m evasion.standalone_indicators --dataset UTKFACE > evasion/results/utkface_CE.txt
# python -m evasion.c35_indicators --dataset UTKFACE > evasion/results/c35_utkface_CE.txt
# python -m evasion.c12_indicators --dataset UTKFACE > evasion/results/c12_utkface_CE.txt
# python -m evasion.c9_indicators --dataset UTKFACE > evasion/results/c9_utkface_CE.txt
