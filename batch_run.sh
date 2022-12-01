#!/bin/bash

models='GraphSage GAT GCN'
datasets='Cora CiteSeer PubMed'
attack_models='GAT GCN'

for attack_model in ${attack_models}; do
    logdir=$(pwd)/log/$(date +%y-%m-%d-%H%M%S)-${attack_model}
    mkdir -p ${logdir}
    for dataset in ${datasets}; do
        for target_model in ${models}; do
            for shadow_model in ${models}; do
                cur_time=$(date +%y%m%d-%H:%M:%S)
                logfile="${logdir}/${dataset}-${target_model}-${shadow_model}-${attack_model}-${cur_time}.log"
                echo ${logfile}
                echo ${dataset} ${target_model} ${shadow_model} ${attack_model} ${cur_time};
                python main.py \
                    --dataset ${dataset}\
                    --target_model ${target_model} \
                    --shadow_model ${shadow_model} \
                    --attack_model ${attack_model} \
                    --logfile_name ${logfile};
            done
        done
    done
    bash parse_log.sh ${logdir}
done
