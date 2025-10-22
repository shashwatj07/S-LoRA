#!/bin/bash
ranks=(8)
rps_list=(7 9 10 12)

for rank in "${ranks[@]}"; do
    for rps in "${rps_list[@]}"; do
        echo "rank=$rank rps=$rps"
        python generate_trace.py -m huggyllama/llama-7b -r ${rank}=1 --rps ${rps} --distribution uniform -t 180 -o ./operating_point_traces
        mv operating_point_traces/uniform_uniform_${rps}.0_180.csv operating_point_traces_sampled/${rank}_rps${rps}_180s.csv
    done
done