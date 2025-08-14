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

python -m pairwise.c9_evasnRobIn_ModelWMPost --dataset FMNIST > results/pairwise/c9_evasnRobIn_ModelWMPost_fmnist.txt
python -m pairwise.c9_evasnRobIn_ModelWMPost --dataset UTKFACE > results/pairwise/c9_evasnRobIn_ModelWMPost_utkface.txt

#autopgd
python -m pairwise.c9_auto_evasnRobIn_ModelWMPost --dataset FMNIST > results/pairwise/c9_auto_evasnRobIn_ModelWMPost_fmnist.txt
python -m pairwise.c9_auto_evasnRobIn_ModelWMPost --dataset UTKFACE > results/pairwise/c9_auto_evasnRobIn_ModelWMPost_utkface.txt

python -m pairwise.c10_PoisonRobIn_fngprntPost --dataset FMNIST > results/pairwise/c10_PoisonRobIn_fngprntPost_fmnist.txt
python -m pairwise.c10_PoisonRobIn_fngprntPost --dataset UTKFACE > results/pairwise/c10_PoisonRobIn_fngprntPost_utkface.txt

python -m pairwise.c11_PoisonRobPost_fngrprntPost --dataset FMNIST > results/pairwise/c11_PoisonRobPost_fngrprntPost_fmnist.txt
python -m pairwise.c11_PoisonRobPost_fngrprntPost --dataset UTKFACE > results/pairwise/c11_PoisonRobPost_fngrprntPost_utkface.txt

python -m pairwise.c12_evasionRobIn_expl --dataset FMNIST > results/pairwise/c12_evasionRobIn_expl_fmnist.txt
python -m pairwise.c12_evasionRobIn_expl --dataset UTKFACE > results/pairwise/c12_evasionRobIn_expl_utkface.txt

#autopgd
python -m pairwise.c12_auto_evasionRobIn_expl --dataset FMNIST > results/pairwise/c12_auto_evasionRobIn_expl_fmnist.txt
python -m pairwise.c12_auto_evasionRobIn_expl --dataset UTKFACE > results/pairwise/c12_auto_evasionRobIn_expl_utkface.txt

python -m pairwise.c13_gpfairIn_PoisonRobPost > results/pairwise/c13_gpfairIn_PoisonRobPost_utkface.txt

python -m pairwise.c14_gpfairIn_ModelWMPre > results/pairwise/c14_gpfairIn_ModelWMPre_utkface.txt

python -m pairwise.c15_gpfairIn_ModelWMPost > results/pairwise/c15_gpfairIn_ModelWMPost_utkface.txt

python -m pairwise.c16_DataWMPre_gpfairIn > results/pairwise/c16_DataWMPre_gpfairIn_utkface.txt

python -m pairwise.c17_gpfairIn_fngprnt > results/pairwise/c17_gpfairIn_fngprnt_utkface.txt

python -m pairwise.c18_gpfairIn_expl > results/pairwise/c18_gpfairIn_expl_utkface.txt

python -m pairwise.c19_poisonRobIn_ModelWMPost --dataset FMNIST > results/pairwise/c19_poisonRobIn_ModelWMPost_fmnist.txt
python -m pairwise.c19_poisonRobIn_ModelWMPost --dataset UTKFACE > results/pairwise/c19_poisonRobIn_ModelWMPost_utkface.txt

python -m pairwise.c20_ModelWMPost_explPost --dataset FMNIST > results/pairwise/c20_ModelWMPost_explPost_fmnist.txt
python -m pairwise.c20_ModelWMPost_explPost --dataset UTKFACE > results/pairwise/c20_ModelWMPost_explPost_utkface.txt

python -m pairwise.c21_poisonRobIn_ModelWMPre --dataset FMNIST > results/pairwise/c21_poisonRobIn_ModelWMPre_fmnist.txt
python -m pairwise.c21_poisonRobIn_ModelWMPre --dataset UTKFACE > results/pairwise/c21_poisonRobIn_ModelWMPre_utkface.txt

python -m pairwise.c22_DataWMPre_ModelWMIn --dataset FMNIST > results/pairwise/c22_DataWMPre_ModelWMIn_fmnist.txt
python -m pairwise.c22_DataWMPre_ModelWMIn --dataset UTKFACE > results/pairwise/c22_DataWMPre_ModelWMIn_utkface.txt # to fix

python -m pairwise.c23_poisonRobPost_DataWMPre --dataset FMNIST > results/pairwise/c23_poisonRobPost_DataWMPre_fmnist.txt
python -m pairwise.c23_poisonRobPost_DataWMPre --dataset UTKFACE > results/pairwise/c23_poisonRobPost_DataWMPre_utkface.txt

python -m pairwise.c24_ModelWMPre_explPost --dataset FMNIST > results/pairwise/c24_ModelWMPre_explPost_fmnist.txt
python -m pairwise.c24_ModelWMPre_explPost --dataset UTKFACE > results/pairwise/c24_ModelWMPre_explPost_utkface.txt

python -m pairwise.c25_ModelWMIn_explPost --dataset FMNIST > results/pairwise/c25_ModelWMIn_explPost_fmnist.txt
python -m pairwise.c25_ModelWMIn_explPost --dataset UTKFACE > results/pairwise/c25_ModelWMIn_explPost_utkface.txt

python -m pairwise.c26_DataWMPre_expl --dataset FMNIST > results/pairwise/c26_DataWMPre_expl_fmnist.txt
python -m pairwise.c26_DataWMPre_expl --dataset UTKFACE > results/pairwise/c26_DataWMPre_expl_utkface.txt

python -m pairwise.c27_poisonRobIn_expl --dataset FMNIST > results/pairwise/c27_poisonRobIn_expl_fmnist.txt
python -m pairwise.c27_poisonRobIn_expl --dataset UTKFACE > results/pairwise/c27_poisonRobIn_expl_utkface.txt

python -m pairwise.c28_poisonRobPost_explPost --dataset FMNIST > results/pairwise/c28_poisonRobPost_explPost_fmnist.txt
python -m pairwise.c28_poisonRobPost_explPost --dataset UTKFACE > results/pairwise/c28_poisonRobPost_explPost_utkface.txt

python -m pairwise.c29_explPost_fngprntPost --dataset FMNIST > results/pairwise/c29_explPost_fngprntPost_fmnist.txt
python -m pairwise.c29_explPost_fngprntPost --dataset UTKFACE > results/pairwise/c29_explPost_fngprntPost_utkface.txt

python -m pairwise.c30_DataWMPre_fngprntPost --dataset FMNIST > results/pairwise/c30_DataWMPre_fngprntPost_fmnist.txt
python -m pairwise.c30_DataWMPre_fngprntPost --dataset UTKFACE > results/pairwise/c30_DataWMPre_fngprntPost_utkface.txt

python -m pairwise.c31_dpIn_ModelWMPost --dataset FMNIST > results/pairwise/c31_dpIn_ModelWMPost_fmnist.txt
python -m pairwise.c31_dpIn_ModelWMPost --dataset UTKFACE > results/pairwise/c31_dpIn_ModelWMPost_utkface.txt

python -m pairwise.c32_DataWMPre_ModelWMPost --dataset FMNIST > results/pairwise/wc32_DataWMPre_ModelWMPost_fmnist.txt
python -m pairwise.c32_DataWMPre_ModelWMPost --dataset UTKFACE > results/pairwise/c32_DataWMPre_ModelWMPost_utkface.txt

python -m pairwise.c33_poisonRobPost_ModelWMPost --dataset FMNIST > results/pairwise/c33_poisonRobPost_ModelWMPost_fmnist.txt
python -m pairwise.c33_poisonRobPost_ModelWMPost --dataset UTKFACE > results/pairwise/c33_poisonRobPost_ModelWMPost_utkface.txt

python -m pairwise.c34_DataWMPre_ModelWMPre --dataset FMNIST > results/pairwise/c34_DataWMPre_ModelWMPre_fmnist.txt
python -m pairwise.c34_DataWMPre_ModelWMPre --dataset UTKFACE > results/pairwise/c34_DataWMPre_ModelWMPre_utkface.txt

python -m pairwise.c35_evasnRobIn_PoisonRobPost --dataset FMNIST > results/pairwise/c35_evasnRobIn_PoisonRobPost_fmnist.txt
python -m pairwise.c35_evasnRobIn_PoisonRobPost --dataset UTKFACE > results/pairwise/c35_evasnRobIn_PoisonRobPost_utkface.txt

#autopgd
python -m pairwise.c35_auto_evasnRobIn_PoisonRobPost --dataset FMNIST > results/pairwise/c35_auto_evasnRobIn_PoisonRobPost_fmnist.txt
python -m pairwise.c35_auto_evasnRobIn_PoisonRobPost --dataset UTKFACE > results/pairwise/c35_auto_evasnRobIn_PoisonRobPost_utkface.txt

python -m pairwise.c36_ModelWMPre_poisonRobIn --dataset FMNIST > results/pairwise/c36_ModelWMPre_poisonRobIn_fmnist.txt
python -m pairwise.c36_ModelWMPre_poisonRobIn --dataset UTKFACE > results/pairwise/c36_ModelWMPre_poisonRobIn_utkface.txt

python -m pairwise.c37_modelWMPre_PoisonRobPost --dataset FMNIST > results/pairwise/c37_modelWMPre_PoisonRobPost_fmnist.txt
python -m pairwise.c37_modelWMPre_PoisonRobPost --dataset UTKFACE > results/pairwise/c37_modelWMPre_PoisonRobPost_utkface.txt

python -m pairwise.c38_ModelWMIn_PoisonRobPost --dataset FMNIST > results/pairwise/c38_ModelWMIn_PoisonRobPost_fmnist.txt
python -m pairwise.c38_ModelWMIn_PoisonRobPost --dataset UTKFACE > results/pairwise/c38_ModelWMIn_PoisonRobPost_utkface.txt
