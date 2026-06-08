#!/bin/bash
set -x

rps_list=(92 96 100 104)
num_servers=12
# backends=("baseline" "contiguous" "system")
backends=("toppings")

python ../run_exp_new.py --trace-file-path ../warmup.csv --servers "http://127.0.0.1:8000" --output warmup.txt --backend baseline --warmup-time 0 --warmup-requests 0

for rps in "${rps_list[@]}"; do
    duration=600
    trace_file_path="../server_maps/uniform_poisson_${rps}.0_${duration}_150_adapters_server_maps_${num_servers}_servers/toppings/uniform_poisson_${rps}.0_${duration}.csv"

    for backend in "${backends[@]}"; do
        mkdir -p ../../../outputs/scalability_exp5h/uniform_poisson_${rps}.0_${duration}_${num_servers}_servers/${backend}

        for server_id in $(seq 1 $num_servers); do
            echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] qps=$rps distribution=uniform arrival=poisson backend=$backend server_idx=$server_id" >> ../../../outputs/run_log_scalability_12_nodes.txt

            python ../run_exp_from_servermaps.py --backend ${backend} --trace-file-path ${trace_file_path} --servers "http://10.0.0.${server_id}:8000" --output uniform_poisson_${rps}.0_${duration}_${num_servers}_servers_server_${server_id}.txt --warmup-time 0 --warmup-requests 0
            rc=$?
            if [ $rc -ne 0 ]; then
                echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] FAILED qps=$rps distribution=uniform arrival=poisson backend=$backend server_idx=$server_id exit_code=${rc}" >> ../../../outputs/run_log_scalability_12_nodes.txt
            fi

            mv uniform_poisson_${rps}.0_${duration}_${num_servers}_servers_server_${server_id}.txt ../../../outputs/scalability_exp5h/uniform_poisson_${rps}.0_${duration}_${num_servers}_servers/${backend}/uniform_poisson_${rps}.0_${duration}_${num_servers}_servers_server_${server_id}.txt
            mv fine_uniform_poisson_${rps}.0_${duration}_${num_servers}_servers_server_${server_id}.txt ../../../outputs/scalability_exp5h/uniform_poisson_${rps}.0_${duration}_${num_servers}_servers/${backend}/fine_uniform_poisson_${rps}.0_${duration}_${num_servers}_servers_server_${server_id}.txt

        done
    done

done