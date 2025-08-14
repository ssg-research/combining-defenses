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

mkdir -p "results/hyperparams"
mkdir -p "results/hyperparams/c17"
mkdir -p "results/hyperparams/c21"
mkdir -p "results/hyperparams/c23"
mkdir -p "results/hyperparams/c32"
mkdir -p "results/hyperparams/c35"
mkdir -p "results/hyperparams/c36"
mkdir -p "results/hyperparams/c37"
mkdir -p "results/hyperparams/c38"

datasets=("FMNIST" "UTKFACE")
trigger_sizes=(3 5)
poisoning_rates=(0.1 0.2 0.3)
wm_rates=(0.1 0.2 0.3)

# C17: Group Fairness + Fingerprinting
# --lamda {0.5, 1, 1.5, 2} 
# --num_iter {25, 50, 75, 100} 
# --val_size {100, 500, 1000}
lamdas=(0.5 1 1.5 2)
iters=(25 50 75 100)
val_sizes=(100 500 1000)
log_dir="results/hyperparams/c17"
for lamda in "${lamdas[@]}"; do
  for iter in "${iters[@]}"; do
    for val_size in "${val_sizes[@]}"; do
      log_file="${log_dir}/lamda${lamda}_iter${iter}_val${val_size}.log"
      echo "Running: lamda=$lamda num_iter=$iter val_size=$val_size"
      python -m pairwise.c17_gpfairIn_fngprnt \
        --lamda "$lamda" \
        --num_iter "$iter" \
        --val_size "$val_size" \
        > "$log_file" 2>&1
    done
  done
done

# C21: Poison Robustness (Intrain) + Model Watermarking (Pretrain)
# --trigger_size {3,5}
# --poisoning_rate {0.1, 0.2, 0.3}
log_dir="results/hyperparams/c21"
for dataset in "${datasets[@]}"; do
  for trigger_size in "${trigger_sizes[@]}"; do
    for poisoning_rate in "${poisoning_rates[@]}"; do
      log_file="${log_dir}/${dataset}_ts${trigger_size}_pr${poisoning_rate}.log"
      echo "Running: $dataset trigger_size=$trigger_size poisoning_rate=$poisoning_rate"
      python -m pairwise.c21_poisonRobIn_ModelWMPre \
        --dataset "$dataset" \
        --trigger_size "$trigger_size" \
        --poisoning_rate "$poisoning_rate" \
        > "$log_file" 2>&1
    done
  done
done

# C23: Poison Robustness (Post) + Data watermarking (Pretrain)
# --trigger_size {3,5}
# --poisoning_rate {0.1, 0.2, 0.3}
log_dir="results/hyperparams/c23"
for dataset in "${datasets[@]}"; do
  for trigger_size in "${trigger_sizes[@]}"; do
    for poisoning_rate in "${poisoning_rates[@]}"; do
      log_file="${log_dir}/${dataset}_ts${trigger_size}_pr${poisoning_rate}.log"
      echo "Running: $dataset trigger_size=$trigger_size poisoning_rate=$poisoning_rate"
      python -m pairwise.c23_poisonRobPost_DataWMPre \
        --dataset "$dataset" \
        --trigger_size "$trigger_size" \
        --poisoning_rate "$poisoning_rate" \
        > "$log_file" 2>&1
    done
  done
done

# C32: Data watermarking (Pretrain) + Model Watermark (Post)
# --alpha {0.002, 0.01, 0.02}
# --wm_rate {0.1, 0.2, 0.3}
# --trigger_size {3,5}
alphas=(0.002 0.01 0.02)
log_dir="results/hyperparams/c32"
for dataset in "${datasets[@]}"; do
  for alpha in "${alphas[@]}"; do
    for wm_rate in "${wm_rates[@]}"; do
      for trigger_size in "${trigger_sizes[@]}"; do
        log_file="${log_dir}/${dataset}_alpha${alpha}_wm${wm_rate}_ts${trigger_size}.log"
        echo "Running: $dataset alpha=$alpha wm_rate=$wm_rate trigger_size=$trigger_size"
        python -m pairwise.c32_DataWMPre_ModelWMPost \
          --dataset "$dataset" \
          --alpha "$alpha" \
          --wm_rate "$wm_rate" \
          --trigger_size "$trigger_size" \
          > "$log_file" 2>&1
      done
    done
  done
done

# C35: Evasion Robustness (In) + Poison Robustness (Post)
# --beta {2,4,6,8}
# --wm_rate {0.1, 0.2, 0.3}
# --trigger_size {3,5}
betas=(2 4 6 8)
log_dir="results/hyperparams/c35"
for dataset in "${datasets[@]}"; do
  for beta in "${betas[@]}"; do
    for wm_rate in "${wm_rates[@]}"; do
      for trigger_size in "${trigger_sizes[@]}"; do
        log_file="${log_dir}/${dataset}_beta${beta}_wm${wm_rate}_ts${trigger_size}.log"
        
        echo "Running: $dataset beta=$beta wm_rate=$wm_rate trigger_size=$trigger_size"
        python -m pairwise.c35_evasnRobIn_PoisonRobPost \
          --dataset "$dataset" \
          --beta "$beta" \
          --wm_rate "$wm_rate" \
          --trigger_size "$trigger_size" \
          > "$log_file" 2>&1
      done
    done
  done
done


# C36: Model Watermark (Pretrain) + Poison Robustness (Intrain)
# --wm_rate {0.1, 0.2, 0.3}
# --trigger_size {3,5}
# --poisoning_rate {0.1, 0.2, 0.3}
log_dir="results/hyperparams/c36"
for dataset in "${datasets[@]}"; do
  for wm_rate in "${wm_rates[@]}"; do
    for trigger_size in "${trigger_sizes[@]}"; do
      for poisoning_rate in "${poisoning_rates[@]}"; do
        log_file="${log_dir}/${dataset}_wm${wm_rate}_ts${trigger_size}_pr${poisoning_rate}.log"
        echo "Running: $dataset wm_rate=$wm_rate trigger_size=$trigger_size poisoning_rate=$poisoning_rate"
        python -m pairwise.c36_ModelWMPre_poisonRobIn \
          --dataset "$dataset" \
          --wm_rate "$wm_rate" \
          --trigger_size "$trigger_size" \
          --poisoning_rate "$poisoning_rate" \
          > "$log_file" 2>&1
      done
    done
  done
done


# C37: Model Watermark (Pretrain) + Poison Robustness (Post)
# --wm_rate {0.1, 0.2, 0.3}
# --trigger_size {3,5}
# --poisoning_rate {0.1, 0.2, 0.3}
log_dir="results/hyperparams/c37"
for dataset in "${datasets[@]}"; do
  for wm_rate in "${wm_rates[@]}"; do
    for trigger_size in "${trigger_sizes[@]}"; do
      for poisoning_rate in "${poisoning_rates[@]}"; do
        log_file="${log_dir}/${dataset}_wm${wm_rate}_ts${trigger_size}_pr${poisoning_rate}.log"
        echo "Running: $dataset wm_rate=$wm_rate trigger_size=$trigger_size poisoning_rate=$poisoning_rate"
        python -m pairwise.c37_modelWMPre_PoisonRobPost \
          --dataset "$dataset" \
          --wm_rate "$wm_rate" \
          --trigger_size "$trigger_size" \
          --poisoning_rate "$poisoning_rate" \
          > "$log_file" 2>&1
      done
    done
  done
done

# C38: Data watermarking (Intrain) + Poison Robustness (Post)
# --wm_rate {0.1, 0.2, 0.3}
# --robust_noise {0.5, 0.75, 1.0, 1.25}
# --robust_noise_step {0.05, 0.10, 0.15}
robust_noises=(0.5 0.75 1.0 1.25)
robust_noise_steps=(0.05 0.10 0.15)
log_dir="results/hyperparams/c38"
for dataset in "${datasets[@]}"; do
  for wm_rate in "${wm_rates[@]}"; do
    for robust_noise in "${robust_noises[@]}"; do
      for robust_noise_step in "${robust_noise_steps[@]}"; do
        log_file="${log_dir}/${dataset}_wm${wm_rate}_rn${robust_noise}_rns${robust_noise_step}.log"
        echo "Running: dataset=$dataset wm_rate=$wm_rate robust_noise=$robust_noise robust_noise_step=$robust_noise_step"
        python -m pairwise.c38_ModelWMIn_PoisonRobPost \
          --dataset "$dataset" \
          --wm_rate "$wm_rate" \
          --robust_noise "$robust_noise" \
          --robust_noise_step "$robust_noise_step" \
          > "$log_file" 2>&1
      done
    done
  done
done