#!/bin/bash
set -x
servers="http://10.0.0.8:8000 http://10.0.0.10:8000 http://10.0.0.4:8000 http://10.0.0.5:8000"
# qps_list=(7 8 9 10)
qps_list=(24 26 28 30 32 34)
distributions=("uniform" "skew")
backends=("baseline" "contiguous" "system")
arrivals=("uniform" "poisson" "bursty")
duration=900 # 15 minutes
model="huggyllama/llama-7b"

python run_exp_new.py --trace-file-path warmup.csv --servers ${servers} --output warmup.txt --backend baseline --warmup-time 0 --warmup-requests 0

: > ../../outputs/run_log.txt

for qps in "${qps_list[@]}"; do
    for distribution in "${distributions[@]}"; do
        for arrival in "${arrivals[@]}"; do
            for backend in "${backends[@]}"; do
                echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] qps=$qps distribution=$distribution arrival=$arrival backend=$backend" >> ../../outputs/run_log.txt

                if [ "$backend" = "system" ]; then
                    warmup_requests=$((360 * qps))
                    python run_exp_new.py --trace-file-path experiment_traces_v2/${distribution}_${arrival}_${qps}.0_${duration}.csv --servers ${servers} --output ${distribution}_${arrival}_${qps}.0_${duration}_${backend}.txt --backend ${backend} --warmup-time 360 --warmup-requests ${warmup_requests}
                    rc=$?
                    if [ $rc -ne 0 ]; then
                        echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] FAILED qps=${qps} distribution=${distribution} arrival=${arrival} backend=${backend} exit_code=${rc}" >> ../../outputs/run_log.txt
                    fi
                else
                    warmup_requests=$((60 * qps))
                    python run_exp_new.py --trace-file-path experiment_traces_v2/${distribution}_${arrival}_${qps}.0_${duration}.csv --servers ${servers} --output ${distribution}_${arrival}_${qps}.0_${duration}_${backend}.txt --backend ${backend} --warmup-time 60 --warmup-requests ${warmup_requests}
                    rc=$?
                    if [ $rc -ne 0 ]; then
                        echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] FAILED qps=${qps} distribution=${distribution} arrival=${arrival} backend=${backend} exit_code=${rc}" >> ../../outputs/run_log.txt
                    fi
                fi
                mv allocation_log.txt ${distribution}_${arrival}_${qps}.0_${duration}_${backend}_allocation_log.txt
                mv server_map.json ${distribution}_${arrival}_${qps}.0_${duration}_${backend}_server_map.json
                mv demand_tps.json ${distribution}_${arrival}_${qps}.0_${duration}_demand_tps.json

                mkdir -p ../../outputs/${distribution}_${arrival}_${qps}/${backend}
                cp ${distribution}_${arrival}_${qps}.0_${duration}_${backend}.txt ../../outputs/${distribution}_${arrival}_${qps}/${backend}
                cp fine_${distribution}_${arrival}_${qps}.0_${duration}_${backend}.txt ../../outputs/${distribution}_${arrival}_${qps}/${backend}
                cp ${distribution}_${arrival}_${qps}.0_${duration}_${backend}_allocation_log.txt ../../outputs/${distribution}_${arrival}_${qps}/${backend}
                cp ${distribution}_${arrival}_${qps}.0_${duration}_${backend}_server_map.json ../../outputs/${distribution}_${arrival}_${qps}/${backend}
                cp ${distribution}_${arrival}_${qps}.0_${duration}_demand_tps.json ../../outputs/${distribution}_${arrival}_${qps}/${backend}
            done
        done
    done
done