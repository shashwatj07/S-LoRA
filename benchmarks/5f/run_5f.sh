#!/bin/bash
set -x
servers="http://10.0.0.8:8000 http://10.0.0.10:8000 http://10.0.0.4:8000 http://10.0.0.5:8000"
qps=28
alphas=(-2 -1 0 1 2)
distributions=("powerlaw")
backends=("baseline" "contiguous" "system")
arrivals=("poisson")
duration=900 # 15 minutes
model="huggyllama/llama-7b"

python ../run_exp_new.py --trace-file-path ../warmup.csv --servers ${servers} --output warmup.txt --backend baseline --warmup-time 0 --warmup-requests 0

: > ../../../outputs/run_log.txt

for alpha in "${alphas[@]}"; do
    for distribution in "${distributions[@]}"; do
        for arrival in "${arrivals[@]}"; do
            for backend in "${backends[@]}"; do
                echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] qps=$qps distribution=$distribution arrival=$arrival backend=$backend alpha=$alpha" >> ../../../outputs/run_log_exp5f.txt

                warmup_requests=$((300 * qps))
                python ../run_exp_new.py --trace-file-path ${distribution}_${arrival}_${qps}.0_${duration}_alpha${alpha}.csv --servers ${servers} --output ${distribution}_${arrival}_${qps}.0_${duration}_alpha${alpha}_${backend}.txt --backend ${backend} --warmup-time 60 --warmup-requests ${warmup_requests}
                rc=$?
                if [ $rc -ne 0 ]; then
                    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] FAILED qps=${qps} distribution=${distribution} arrival=${arrival} backend=${backend} alpha=${alpha} exit_code=${rc}" >> ../../../outputs/run_log.txt
                fi

                mv allocation_log.txt ${distribution}_${arrival}_${qps}.0_${duration}_alpha${alpha}_${backend}_allocation_log.txt
                mv server_map.json ${distribution}_${arrival}_${qps}.0_${duration}_alpha${alpha}_${backend}_server_map.json
                mv demand_tps.json ${distribution}_${arrival}_${qps}.0_${duration}_alpha${alpha}_${backend}_demand_tps.json

                mkdir -p ../../../outputs/${distribution}_${arrival}_${qps}_alpha${alpha}/${backend}
                cp ${distribution}_${arrival}_${qps}.0_${duration}_alpha${alpha}_${backend}.txt ../../../outputs/${distribution}_${arrival}_${qps}_alpha${alpha}/${backend}
                cp fine_${distribution}_${arrival}_${qps}.0_${duration}_alpha${alpha}_${backend}.txt ../../../outputs/${distribution}_${arrival}_${qps}_alpha${alpha}/${backend}
                cp ${distribution}_${arrival}_${qps}.0_${duration}_alpha${alpha}_${backend}_allocation_log.txt ../../../outputs/${distribution}_${arrival}_${qps}_alpha${alpha}/${backend}
                cp ${distribution}_${arrival}_${qps}.0_${duration}_alpha${alpha}_${backend}_server_map.json ../../../outputs/${distribution}_${arrival}_${qps}_alpha${alpha}/${backend}
                cp ${distribution}_${arrival}_${qps}.0_${duration}_alpha${alpha}_${backend}_demand_tps.json ../../../outputs/${distribution}_${arrival}_${qps}_alpha${alpha}/${backend}
            done
        done
    done
done