#!/bin/bash
set -x

qps_list=(20 24)
# distributions=("uniform" "skew")
distributions=("skewslide")
arrivals=("uniform" "poisson" "bursty")
# arrivals=("bursty")
duration=900 # 15 minutes
model="huggyllama/llama-7b"
output_dir="./experiment_traces_v2"

for qps in "${qps_list[@]}"; do
    for distribution in "${distributions[@]}"; do
        for arrival in "${arrivals[@]}"; do
            echo "qps=$qps distribution=$distribution arrival=$arrival"
            python generate_trace.py -m ${model} -r 8=5 16=5 32=5 64=5 128=5 --rps ${qps} --distribution ${distribution} --arrival ${arrival} -t ${duration} -o ${output_dir}
        done
    done
done