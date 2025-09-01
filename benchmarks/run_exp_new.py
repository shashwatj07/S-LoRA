"""
To run with real mode:
python run_exp.py --backend dm --suite a10g --breakdown  --mode real
with synthetic mode:
python run_exp.py --backend dm --suite a10g --breakdown  --mode synthetic
default to synthetic mode.
"""
import argparse
import asyncio
import csv
import json
import re
import numpy as np
import os
import sys
import time
import random
import bisect
from tqdm import tqdm
from typing import List, Tuple

import aiohttp

from exp_suite import BenchmarkConfig, get_all_suites, to_dict, BASE_MODEL, LORA_DIR
from trace import generate_requests, get_real_requests, Request, dummy_prompt
sys.path.append("../bench_lora")
from dancingmodel.utils.metric import reward, attainment_func

GB = 1024 ** 3

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []
vllm_packed_adapter_dir_to_url_map = {}


def get_peak_mem(server):
    url = server + "/get_peak_mem"
    response = requests.post(url)
    return response.json()["peak_mem"]


async def send_request(
    backend: str,
    server: str,
    req_id: str,
    model_dir: str,
    adapter_dir: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    output_file: str,
    debug: bool,
) -> None:
    request_start_time = time.time()
    headers = {'Content-Type': 'application/json'}
    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        url = server + "/generate"
    elif backend == "vllm-packed":
        url = vllm_packed_adapter_dir_to_url_map[adapter_dir] + "/generate"
    else:
        url = server + "/generate_stream"
    
    if backend in ["dm", "system", "baseline", "contiguous"]:
        data = {
            'model_dir': model_dir,
            'lora_dir': adapter_dir,
            'inputs': prompt,
            'parameters': {
                'do_sample': False,
                'ignore_eos': True,
                'max_new_tokens': output_len,
                 # 'temperature': 0.1,
            }
        }
    elif backend in ["lightllm"]:
        data = {
            'inputs': prompt,
            'parameters': {
                'do_sample': False,
                'ignore_eos': True,
                'max_new_tokens': output_len,
                 # 'temperature': 0.1,
            },
        }
    elif backend in ["vllm", "vllm-packed"]:
        data = {
            'prompt': prompt,
            'max_tokens': output_len,
            'ignore_eos': True,
        }
    # with open(f"fine_{output_file}", "a") as f:
    #     f.write(f"sent {req_id}, {server}\n")
    first_token_latency = None
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        while True:
            async with session.post(url, headers=headers, json=data) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    if first_token_latency is None:
                        first_token_latency = time.time() - request_start_time
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            # output = json.loads(output)
            # print(output)
            
            if '\"finished\": -1' not in output:
                break
            else:
                first_token_latency = None
                break
            # #     print(output)
            # #     print(json.loads(output))
            # break

    request_end_time = time.time()
    request_latency = request_end_time - request_start_time
    tbt = (request_latency - first_token_latency) / (output_len - 1) if output_len > 1 else 0
    log_line = f"req_id {req_id} {server} adapter_dir {adapter_dir} prompt_len {prompt_len} output_len {output_len} "\
          f"request_latency {request_latency:.2f} s, first_token_latency {first_token_latency:.2f} s, tbt {tbt:.2f} s\n"
    print(log_line)
    with open(f"fine_{output_file}", "a") as f:
        f.write(log_line)
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency, first_token_latency, tbt))
    return (prompt_len, output_len, request_latency, first_token_latency, tbt)

async def benchmark_baseline(
    backend: str,
    server_map: str,
    input_requests: List[Tuple[str, str, str, int, int]],
    output,
    debug=False,
) -> None:
    start = time.time()
    tasks: List[asyncio.Task] = []
    for req in input_requests:
        await asyncio.sleep(start + req.req_time - time.time())
        if debug:
            print(f"{req.req_id} {req.req_time:.5f} wait {start + req.req_time - time.time():.5f} "
                  f"{req.adapter_dir}")
        # print(req)
        
        task = asyncio.create_task(send_request(backend, server_map[req.adapter_dir],
                                                req.req_id, req.model_dir, req.adapter_dir, req.prompt,
                                                req.prompt_len, req.output_len, output, debug))
        tasks.append(task)
    latency = await asyncio.gather(*tasks)
    return latency

async def benchmark_system(
    backend: str,
    server_map: str,
    adapter_dirs,
    input_requests: List[Tuple[str, str, str, int, int]],
    output,
    debug=False,
) -> None:
    start = time.time()
    tasks: List[asyncio.Task] = []
    last_time = input_requests[0]
    step = 30
    for req in input_requests:
        # print(req.req_id)
        if req.req_time > last_time.req_time // 1 + step:
            demand_tps = {a:0 for a in adapter_dirs}
            index = input_requests.index(last_time)
            while input_requests[index].req_time < req.req_time:
                # rank = int(re.search(r'rank-(\d+)', input_requests[index].adapter_dir).group(1))
                demand_tps[input_requests[index].adapter_dir] = demand_tps.get(input_requests[index].adapter_dir) + (input_requests[index].prompt_len + input_requests[index].output_len) / step
                index += 1
            # print(demand_tps)
            adapter_demand = []
            for adapter, tps in demand_tps.items():
                rank = int(re.search(r'rank-(\d+)', adapter).group(1))
                adapter_demand.append((rank, tps, adapter)) # tps here is expected tps
            adapter_demand.sort(reverse=True)

            # server_tps = {8: 2400, 16: 2100, 32: 1900, 64:1700, 128:1600} # operating point, fn of max rank
            server_tps = {8: 2725, 16: 2700, 32: 2675, 64: 2625, 128: 2525} # operating point, fn of max rank
            # TODO score = expected total tps / max rank operating point
            # assume 2000 as operating point of adapter for score of adapters

            #s1: 128 64 32
            #s2: 16 8

            # greedy bin packing
            def is_compatible(group, tuple, scale=1):
                ranks = [r for  r, _, _ in group]
                ranks.append(tuple[0])
                max_rank = max(ranks)
                tps = sum([tps for _, tps, _ in group]) + tuple[1]
                return tps <= (server_tps[max_rank] * scale)
            # def get_tps(adapters, start, end):
            #     adapter_slice = adapters[start:end+1]
            #     ranks = [r for  r, _, _ in adapter_slice]
            #     max_rank = max(ranks)
            #     tps = sum([tps for _, tps, _ in adapter_slice])
            #     return tps / server_tps[max_rank]
            # tps_delta = []
            # for i in range(len(adapter_demand) - 1):
            #     tps1 = get_tps(adapter_demand, 0, i)
            #     tps2 = get_tps(adapter_demand, i+1, len(adapter_demand) - 1)
            #     tps_delta.append(abs(tps1 - tps2))
            # partition_index = tps_delta.index(min(tps_delta))
            # print(partition_index)
            servers = sorted(list(set(server_map.values())))
            assert len(servers) == 2
            # adapter_groups = [adapter_demand[0:partition_index+1], adapter_demand[partition_index+1:]]
            adapter_groups = [[] for _ in servers]
            x = 0
            for i, adapter_tuple in enumerate(adapter_demand):
                if is_compatible(adapter_groups[x], adapter_tuple):
                    adapter_groups[x].append(adapter_tuple)
                elif x + 1 < len(adapter_groups):
                    x += 1
                    adapter_groups[x].append(adapter_tuple)
                else:
                    with open("allocation_log.txt", "a") as f:
                        f.write(f"{last_time} Overallocation for adapters from index {i} to {len(adapter_demand) - 1} ({(len(adapter_demand) - 1 - i + 1) / len(adapter_demand) * 100:.2f} % of total adapters)\n")
                    for j in range(i, len(adapter_demand)):
                        adapter_groups[j % len(adapter_groups)].append(adapter_demand[j])
                    break

            print(adapter_groups)
            with open("allocation_log.txt", "a") as f:
                f.write(f"\n{last_time} Adapter groups:\n")
                for i, group in enumerate(adapter_groups):
                    f.write(f"  Server {servers[i]}: {[(adapter, tps) for _, tps, adapter in group]}\n")
                    f.write(f"  Server {servers[i]} total tps: {sum([tps for _, tps, _ in group])}\n")
                    
            server_map = {}
            for i, server in enumerate(servers):
                for _, _, adapter in adapter_groups[i]:
                    server_map[adapter] = server
            print(server_map)
            last_time = req
        await asyncio.sleep(start + req.req_time - time.time())
        if debug:
            print(f"{req.req_id} {req.req_time:.5f} wait {start + req.req_time - time.time():.5f} "
                  f"{req.adapter_dir}")
        # print(req)
        
        task = asyncio.create_task(send_request(backend, server_map[req.adapter_dir],
                                                req.req_id, req.model_dir, req.adapter_dir, req.prompt,
                                                req.prompt_len, req.output_len, output, debug))
        tasks.append(task)
    latency = await asyncio.gather(*tasks)
    return latency


def get_adapter_dirs(num_adapters, adapter_dirs, backend=None):
    ret = []
    num_iter = num_adapters // len(adapter_dirs) + 1

    if backend == "vllm-packed":
        num_iter = num_adapters // len(adapter_dirs)

    for i in range(num_iter):
        for adapter_dir in adapter_dirs:
            ret.append(adapter_dir + f"-{i}")
    print(ret)
    return ret

def get_res_stats(per_req_latency, benchmark_time, backend, warmup_time=0, warmup_num=0):
    # get throughput
    num_abort = len([i for i in per_req_latency if i[3] is None])
    per_req_latency = [i for i in per_req_latency if i[3] is not None]
    throughput = len(per_req_latency) / benchmark_time
    # print(per_req_latency)
    # if backend == "dm":
    #     peak_mem = get_peak_mem(server)
    #     print(f"GPU peak memory (GB):", [[f"{x / GB:.2f}" for x in tpg] for tpg in peak_mem])
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Non warmup time: {benchmark_time - warmup_time:.2f} s")
    print(f"Total requests: {len(per_req_latency)}")
    print(f"Non warmup requests: {len(per_req_latency) - warmup_num}")
    print(f"Aborted Request: {num_abort}")
    print(f"Throughput: {throughput:.2f} requests/s")

    strip_throughput = (len(per_req_latency) - warmup_num * 2) / (benchmark_time - warmup_time * 2)
    print(f"Throughput strip: {strip_throughput:.2f} requests/s")

    # Exclude first warmup_num requests from further statistics
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    if warmup_num >= len(per_req_latency):
        raise ValueError("Warmup number exceeds number of requests whose latency was logged")
    else:
        stats_base = per_req_latency[warmup_num:]

    e2e_latencies = [latency for _, _, latency, _, _ in stats_base]
    per_token_latencies = [
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency, _, _ in stats_base
        if (prompt_len + output_len) > 0
    ]
    per_output_token_latencies = [
        latency / output_len
        for _, output_len, latency, _, _ in stats_base
        if output_len > 0
    ]
    first_token_latencies = [latency for _, _, _, latency, _ in stats_base]
    tbts = [latency for _, _, _, _, latency in stats_base]
    num_abort = len([i for i in stats_base if i[3] is None])
    abort_satisfaction = [0] * num_abort
    satisfactions = [reward(latency) for _, _, _, latency, _ in stats_base] + abort_satisfaction
    attainments = [attainment_func(latency) for _, _, _, latency, _ in stats_base] + abort_satisfaction

    metrics = {
        "e2e": e2e_latencies,
        "per_token": per_token_latencies,
        "per_output_token": per_output_token_latencies,
        "first_token": first_token_latencies,
        "tbt": tbts,
        "satisfaction": satisfactions,
        "attainment": attainments,
    }

    stats = {}
    for name, values in metrics.items():
        stats[name] = {
            "avg": np.mean(values),
            **{f"p{p}": np.percentile(values, p) for p in percentiles}
        }

    # dump results
    if backend == "dm":
        # TODO
        # single_gpu_peak_mem = peak_mem
        single_gpu_peak_mem = 0
    else:
        single_gpu_peak_mem = 0

    result = {"total_time": benchmark_time, "gpu_peak_mem": single_gpu_peak_mem, "num_abort": num_abort,
              "throughput": throughput, "strip_throughput": strip_throughput,
              "stats": stats, "backend": backend}
    res = {"result": result}
    
    return res

def read_requests(trace_file):
    requests = []
    adapter_dirs = set()
    with open(trace_file, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            elements = line.split(",")
            requests.append(Request(req_id=int(elements[0]), model_dir=elements[1], adapter_dir=elements[2], 
              prompt=dummy_prompt(int(elements[3])), prompt_len=int(elements[3]),
              output_len=int(elements[4]), req_time=float(elements[5])))
            # requests.append((int(elements[0]),elements[1],elements[2],int(elements[3]),int(elements[4]),float(elements[5])))
            adapter_dirs.add(elements[2])
    requests.sort(key=lambda r: r.req_time)
    return list(adapter_dirs), requests

def run_exp(backend, servers, trace_file, output, debug=False, warmup_time:int = 60, warmup_num: int = 600):
    # first generate your data using real_trace/clean_chat_data.py
    # base_model = BASE_MODEL[model_setting]
    # adapter_dirs = LORA_DIR[model_setting]
    adapter_dirs, requests = read_requests(trace_file=trace_file)
    # print(requests)
    avg_prompt_len = np.mean([req.prompt_len for req in requests])
    avg_output_len = np.mean([req.output_len for req in requests])
    avg_len = np.mean([req.prompt_len + req.output_len for req in requests])
    print("num_adapters", len(adapter_dirs), "num_requests", len(requests), "avg_len:", avg_len, "avg_prompt_len:", avg_prompt_len, "avg_output_len:", avg_output_len)
        
    if debug:
        print("num requests:", len(requests))
        for req in requests[:4]:
            print(req)
    if backend == "baseline":
        # benchmark
        random.seed(42)
        shuffled = adapter_dirs.copy()
        random.shuffle(shuffled)
        
        # Split shuffled list into n parts as evenly as possible
        k, m = divmod(len(shuffled), len(servers))
        # server_map = {adapter:server_name for adapter in shuffled[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i, server_name in enumerate(servers)}
        server_map = {}
        for i, server_name in enumerate(servers):
            start = i * k + min(i, m)
            end = (i + 1) * k + min(i + 1, m)
            for adapter in shuffled[start:end]:
                server_map[adapter] = server_name
        print(server_map)
        benchmark_start_time = time.time()
        per_req_latency = asyncio.run(benchmark_baseline(backend, server_map, requests, output, debug))
        benchmark_end_time = time.time()
        benchmark_time = benchmark_end_time - benchmark_start_time
    elif backend == "system" or backend == "contiguous":
        # benchmark
        adapters = []
        for adapter in adapter_dirs:
            rank = int(re.search(r'rank-(\d+)', adapter).group(1))
            adapters.append((rank, adapter))
        adapters.sort()
        adapters = [x[1] for x in adapters[:]]
        
        # Split shuffled list into n parts as evenly as possible
        k, m = divmod(len(adapters), len(servers))
        server_map = {}
        for i, server_name in enumerate(servers):
            start = i * k + min(i, m)
            end = (i + 1) * k + min(i + 1, m)
            for adapter in adapters[start:end]:
                server_map[adapter] = server_name
        print("server map ", server_map)
        benchmark_start_time, benchmark_end_time = 0, 0

        if backend == "system":
            benchmark_start_time = time.time()
            per_req_latency = asyncio.run(benchmark_system(backend, server_map, adapter_dirs, requests, output, debug))
            benchmark_end_time = time.time()
        elif backend == "contiguous":
            benchmark_start_time = time.time()
            per_req_latency = asyncio.run(benchmark_baseline(backend, server_map, requests, output, debug))
            benchmark_end_time = time.time()
        benchmark_time = benchmark_end_time - benchmark_start_time

    res = get_res_stats(per_req_latency, benchmark_time, backend,
                        warmup_time=warmup_time, warmup_num=warmup_num)

    with open(output, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, required=True,
                        choices=["system", "baseline", "contiguous"])

    # parser.add_argument("--model-setting", type=str, default="S1")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--append", action="store_true")

    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--no-lora-compute", action="store_true")
    parser.add_argument("--no-lora-swap", action="store_true")
    parser.add_argument("--no-lora-copy", action="store_true")
    parser.add_argument("--trace-file-path", required=True)

    parser.add_argument("--servers", "-s", type=str, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--warmup-time", type=int, default=60, 
                        help="Number of seconds considered as warmup, excluded from stats")
    parser.add_argument("--warmup-requests", type=int, default=600,
                        help="Number of requests considered as warmup, excluded from stats (rps * warmup-time)")
    args = parser.parse_args()

    assert not args.no_lora_copy or args.no_lora_compute
    assert not (args.debug and args.breakdown)

    # set output file name
    if args.output is None:
        args.output = f"all_results_{args.mode}_" + args.backend + ".jsonl"
    if args.no_lora_swap and args.no_lora_compute and args.no_lora_copy:
        args.output = "no_lora_compute_swap_copy_results.jsonl"
    elif args.no_lora_swap and args.no_lora_compute:
        args.output = "no_lora_compute_swap_results.jsonl"
    elif args.no_lora_swap:
        args.output = "no_lora_swap_results.jsonl"
    elif args.no_lora_compute:
        args.output = "no_lora_compute_results.jsonl"
    if args.debug or args.breakdown:
        args.output = "debug_" + args.output

    # suites = get_all_suites(mode=args.mode, debug=args.debug, suite=args.suite, breakdown=args.breakdown)

    if not args.append:
        os.system(f"rm {args.output}")
        os.system(f"rm allocation_log.txt")
        results = []
    else:
        with open(args.output, "r") as f:
            lines = f.readlines()
        results = [json.loads(line)["config"] for line in lines]

    # for config in tqdm(suites, desc="suites"):
    #     if to_dict(config) not in results:
    stats = run_exp(args.backend, args.servers, args.trace_file_path,
                            args.output, args.debug, args.warmup_time, args.warmup_requests)
