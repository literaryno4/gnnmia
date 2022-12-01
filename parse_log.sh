#!/bin/bash

# parse logs to result file

log_path=$(pwd)/log/$(basename $1)

datasets='Cora CiteSeer PubMed'
models='GraphSage GAT GCN'
attack_model=GraphSage

file_name=results/$(basename $1).result
rm -f ${file_name}
for dataset in ${datasets}; do
    printf "\n\n                                 ${dataset}                               \navg/best" | tee -a ${file_name}
    for shadow_model in ${models}; do
        printf "     ${shadow_model}       " | tee -a ${file_name}
    done
    printf "\n" | tee -a ${file_name}
    for target_model in ${models}; do
        printf "%9s   " ${target_model} | tee -a ${file_name}
        for shadow_model in ${models}; do
            pattern="${dataset}-${target_model}-${shadow_model}-${attack_model}*.log"
            printf " $(sed -n  '/acc/p' $(find ${log_path} -name ${pattern}) | sed 's/\[.*\]\: avg\/best acc\: //g' - | awk 1 ORS=' ' -)" | tee -a ${file_name}
        done
        printf '\n' | tee -a ${file_name}
    done
done