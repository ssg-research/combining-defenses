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


# evasion robustness
python -m standalone.evasion_inprocessing --dataset UTKFACE > ./results/standalone/evasion_inprocessing_utkface.txt
python -m standalone.evasion_inprocessing --dataset FMNIST > ./results/standalone/evasion_inprocessing_fmnist.txt

# evasion robustness (autopgd_origin)
python -m standalone.evasion_inprocessing_auto --dataset UTKFACE > ./results/standalone/evasion_inprocessing_utkface_auto.txt
python -m standalone.evasion_inprocessing_auto --dataset FMNIST > ./results/standalone/evasion_inprocessing_fmnist_auto.txt

# evasion robustness (autopgd_art)
python -m standalone.evasion_inprocessing_art_auto --dataset UTKFACE > ./results/standalone/evasion_inprocessing_utkface_art_auto.txt
python -m standalone.evasion_inprocessing_art_auto --dataset FMNIST > ./results/standalone/evasion_inprocessing_fmnist_art_auto.txt

# evasion robustness (autopgd_ogd)
python -m standalone.evasion_inprocessing_art_pgd --dataset UTKFACE > ./results/standalone/evasion_inprocessing_utkface_art_pgd.txt
python -m standalone.evasion_inprocessing_art_pgd --dataset FMNIST > ./results/standalone/evasion_inprocessing_fmnist_art_pgd.txt

# poison robustness
python -m standalone.poisonrob_inprocessing --dataset UTKFACE > ./results/standalone/poisonrob_inprocessing_utkface.txt
python -m standalone.poisonrob_inprocessing --dataset FMNIST > ./results/standalone/poisonrob_inprocessing_fmnist.txt
python -m standalone.poisonrob_postprocessing --dataset UTKFACE > ./results/standalone/poisonrob_postprocessing_utkface.txt
python -m standalone.poisonrob_postprocessing --dataset FMNIST > ./results/standalone/poisonrob_postprocessing_fmnist.txt

# model watermarking
python -m standalone.modelwm_preprocessing --dataset UTKFACE > ./results/standalone/modelwm_preprocessing_utkface.txt
python -m standalone.modelwm_preprocessing --dataset FMNIST > ./results/standalone/modelwm_preprocessing_fmnist.txt
python -m standalone.modelwm_inprocessing --dataset UTKFACE > ./results/standalone/modelwm_inprocessing_utkface.txt
python -m standalone.modelwm_inprocessing --dataset FMNIST > ./results/standalone/modelwm_inprocessing_fmnist.txt
python -m standalone.modelwm_postprocessing --dataset UTKFACE > ./results/standalone/modelwm_postprocessing_utkface.txt
python -m standalone.modelwm_postprocessing --dataset FMNIST > ./results/standalone/modelwm_postprocessing_fmnist.txt

# dataset watermarking
python -m standalone.datawm_preprocessing --dataset FMNIST > ./results/standalone/datawm_preproc_fmnist.txt
python -m standalone.datawm_preprocessing --dataset UTKFACE > ./results/standalone/datawm_preproc_utkface.txt

# fingerprinting
python -m standalone.fingerprint_postprocessing --dataset FMNIST > ./results/standalone/fngrprnt_postproc_fmnist.txt
python -m standalone.fingerprint_postprocessing --dataset UTKFACE > ./results/standalone/fngrprnt_preproc_utkface.txt

# differential privacy
python -m standalone.dp_inprocessing --dataset UTKFACE > ./results/standalone/dp_inprocessing_utkface.txt
python -m standalone.dp_inprocessing --dataset FMNIST > ./results/standalone/dp_inprocessing_fmnist.txt

# group fairness
python -m standalone.gpfair_inprocessing --dataset UTKFACE > ./results/standalone/gpfair_inprocessing.txt

# explanations
python -m standalone.expl_postprocessing --dataset UTKFACE > ./results/standalone/expl_postprocessing_utkface.txt
python -m standalone.expl_postprocessing --dataset FMNIST > ./results/standalone/expl_postprocessing_fmnist.txt