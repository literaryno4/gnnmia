#!/bin/bash
models='GraphSage GAT GCN'
datasets='Cora CiteSeer PubMed'
attack_mode=GraphSage

for dataset in ${datasets}; do
    for target_model in ${models}; do
        for shadow_model in ${models}; do
            cur_time=$(date +%y%m%d-%H:%M:%S)
            logfile="$(pwd)/log/${dataset}-${target_model}-${shadow_model}-${attack_mode}-${cur_time}.log"
            echo ${logfile}
            echo ${dataset} ${target_model} ${shadow_model} ${attack_mode} ${cur_time};
            python main.py \
                --dataset ${dataset}\
                --target_model ${target_model} \
                --shadow_model ${shadow_model} \
                --attack_model ${attack_mode} \
                --logfile_name ${logfile};
        done
    done
done