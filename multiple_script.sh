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

python -m multiple.advtr_wmModelPost_expl --dataset UTKFACE > results/multiple/advtr_wmModelPost_expl_utkface.txt
python -m multiple.advtr_wmModelPost_expl --dataset FMNIST > results/multiple/advtr_wmModelPost_expl_fmnist.txt

python -m multiple.gpfair_outremPost_expl > results/multiple/gpfair_outremPost_expl_utkface.txt

python -m multiple.wmData_wmModelPost_expl --dataset UTKFACE > results/multiple/wmData_wmModelPost_expl_utkface.txt
python -m multiple.wmData_wmModelPost_expl --dataset FMNIST > results/multiple/wmData_wmModelPost_expl_fmnist.txt

python -m multiple.wmData_gpfair_expl --dataset UTKFACE > results/multiple/wmData_gpfair_expl_utkface.txt

python -m multiple.outremInP_wmModelPost_expl --dataset FMNIST > results/multiple/outremInP_wmModelPost_expl_fmnist.txt
python -m multiple.outremInP_wmModelPost_expl --dataset UTKFACE > results/multiple/outremInP_wmModelPost_expl_utkface.txt

python -m multiple.gpfair_outremPost_wmModelPost > results/multiple/gpfair_outremPost_wmModelPost_utkface.txt

python -m multiple.wmData_gpfair_wmModel > results/multiple/wmData_gpfair_wmModel_utkface.txt

python -m multiple.gpfair_wmModelPost_outremPost_expl > results/multiple/gpfair_wmModelPost_outremPost_expl_utkface.txt

python -m multiple.outremPost_wmModelPost_expl --dataset FMNIST > results/multiple/outremPost_wmModelPost_expl_fmnist.txt
python -m multiple.outremPost_wmModelPost_expl --dataset UTKFACE > results/multiple/outremPost_wmModelPost_expl_utkface.txt


