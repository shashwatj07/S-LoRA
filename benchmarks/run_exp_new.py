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
from itertools import chain
import json
import re
import numpy as np
import os
import sys
import time
import random
import heapq
import bisect
from tqdm import tqdm
from typing import List, Tuple
from math import ceil
from collections import defaultdict, deque
from sklearn.linear_model import LinearRegression

import aiohttp

from exp_suite import BenchmarkConfig, get_all_suites, to_dict, BASE_MODEL, LORA_DIR
from trace import generate_requests, get_real_requests, Request, dummy_prompt

sys.path.append("../bench_lora")
from dancingmodel.utils.metric import reward, attainment_func

GB = 1024**3

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []
vllm_packed_adapter_dir_to_url_map = {}


REMAINING_PREFILLS_PER_SERVER = defaultdict(dict)
REMAINING_DECODES_PER_SERVER = defaultdict(dict)
PREFILL_LOCK = asyncio.Lock()
DECODE_LOCK = asyncio.Lock()


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
    arrival_time: float
) -> None:
    request_start_time = time.time()
    scheduling_latency = max(0.0, request_start_time - arrival_time)
    headers = {"Content-Type": "application/json"}
    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        url = server + "/generate"
    elif backend == "vllm-packed":
        url = vllm_packed_adapter_dir_to_url_map[adapter_dir] + "/generate"
    else:
        url = server + "/generate_stream"

    if backend in ["dm", "system", "baseline", "contiguous", "toppings"]:
        data = {
            "model_dir": model_dir,
            "lora_dir": adapter_dir,
            "inputs": prompt,
            "parameters": {
                "do_sample": False,
                "ignore_eos": True,
                "max_new_tokens": output_len,
                # 'temperature': 0.1,
            },
        }
    elif backend in ["lightllm"]:
        data = {
            "inputs": prompt,
            "parameters": {
                "do_sample": False,
                "ignore_eos": True,
                "max_new_tokens": output_len,
                # 'temperature': 0.1,
            },
        }
    elif backend in ["vllm", "vllm-packed"]:
        data = {
            "prompt": prompt,
            "max_tokens": output_len,
            "ignore_eos": True,
        }
    # with open(f"fine_{output_file}", "a") as f:
    #     f.write(f"sent {req_id}, {server}\n")
    first_token_latency = None
    server_queuing_time, server_execution_time = None, None
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        while True:
            async with session.post(url, headers=headers, json=data) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    if first_token_latency is None:
                        first_token_latency = time.time() - request_start_time
                        if backend == "toppings":
                            rank = int(re.search(r"rank-(\d+)", adapter_dir).group(1))
                            await PREFILL_LOCK.acquire()
                            REMAINING_PREFILLS_PER_SERVER[server][rank] -= prompt_len
                            PREFILL_LOCK.release()
                    
                    if backend == "toppings":
                        rank = int(re.search(r"rank-(\d+)", adapter_dir).group(1))
                        await DECODE_LOCK.acquire()
                        REMAINING_DECODES_PER_SERVER[server][rank] -= 1
                        DECODE_LOCK.release()
                    try:
                        chunk_str = chunk.decode("utf-8")
                        if 'queue_time' in chunk_str:
                            # print(chunk_str)
                            json_str = chunk_str.split("data:", 1)[1]
                            data_obj = json.loads(json_str)
                            server_queuing_time = data_obj['token'].get('queue_time')
                            server_execution_time = data_obj['token'].get('prefill_time')
                    except Exception as e:
                        print(f"Error decoding chunk: {e}")
                        chunk_str = ""
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            # output = json.loads(output)
            # print(output)

            if '"finished": -1' not in output:
                break
            else:
                first_token_latency = None
                break
            # #     print(output)
            # #     print(json.loads(output))
            # break

    request_end_time = time.time()
    request_latency = request_end_time - request_start_time
    tbt = (
        (request_latency - first_token_latency) / (output_len - 1)
        if output_len > 1
        else 0
    )
    log_line = (
        f"req_id {req_id} {server} adapter_dir {adapter_dir} prompt_len {prompt_len} output_len {output_len} "
        f"request_latency {request_latency:.4f} s, client_scheduling_latency {scheduling_latency:.4f} s, first_token_latency {first_token_latency:.4f} s, tbt {tbt:.4f} s"
    )
    
    if server_queuing_time is not None and server_execution_time is not None:
        log_line += f", server_prefill_queuing_time {server_queuing_time:.4f} s, server_prefill_execution_time {server_execution_time:.4f} s"

    print(log_line)
    with open(f"fine_{output_file}", "a") as f:
        f.write(log_line + "\n")
    REQUEST_LATENCY.append(
        (prompt_len, output_len, request_latency, first_token_latency, tbt, scheduling_latency)
    )
    return (prompt_len, output_len, request_latency, first_token_latency, tbt, scheduling_latency)


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
        arrival_time = start + req.req_time
        sleep_time = arrival_time - time.time()
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        if debug:
            print(
                f"{req.req_id} {req.req_time:.5f} wait {start + req.req_time - time.time():.5f} "
                f"{req.adapter_dir}"
            )
        # print(req)

        assert server_map[req.adapter_dir] is not None

        task = asyncio.create_task(
            send_request(
                backend,
                server_map[req.adapter_dir],
                req.req_id,
                req.model_dir,
                req.adapter_dir,
                req.prompt,
                req.prompt_len,
                req.output_len,
                output,
                debug,
                arrival_time
            )
        )
        tasks.append(task)
    latency = await asyncio.gather(*tasks)
    return latency

async def calc_cost_toppings(req, server, operating_points):
    time_to_process = 0.0
    
    for rank in [8, 16, 32, 64, 128]:
        await PREFILL_LOCK.acquire()
        await DECODE_LOCK.acquire()
        remaining_prefills = REMAINING_PREFILLS_PER_SERVER[server].get(rank, 0)
        # remaining_decodes = REMAINING_DECODES_PER_SERVER[server].get(rank, 0)
        # jitter_factor = 1 + random.uniform(-0.25, 0.25) 
        # jittered_decodes = max(0, int(remaining_decodes * jitter_factor))
        # total_tokens_rank = remaining_prefills + jittered_decodes
        avg_decode = 70
        total_tokens_rank = remaining_prefills + avg_decode
        PREFILL_LOCK.release()
        DECODE_LOCK.release()
        time_for_rank = total_tokens_rank / operating_points[rank]
        time_to_process += time_for_rank
        
    return time_to_process

async def benchmark_toppings(
    backend: str,
    input_requests: List[Tuple[str, str, str, int, int]],
    output,
    servers,
    debug=False,
) -> None:
    start = time.time()
    tasks: List[asyncio.Task] = []
    for req in input_requests:
        arrival_time = start + req.req_time
        sleep_time = arrival_time - time.time()
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        if debug:
            print(
                f"{req.req_id} {req.req_time:.5f} wait {start + req.req_time - time.time():.5f} "
                f"{req.adapter_dir}"
            )
        # print(req)
        
        # operating_points = {
        #     8: 5500,
        #     16: 5400,
        #     32: 5250,
        #     64: 5000,
        #     128: 4500,
        # }  # operating point, fn of max rank, ND96asrv 8xA100 80GB
        
        operating_points = {8: 5950, 16: 5150, 32: 4700, 64: 5050, 128: 4650} # prod 7b tp1, NC24ads
        
        #! todo, select server with toppings algo
        chosen_server = servers[0]
        min_cost = np.inf

        for server in servers:
            cost = await calc_cost_toppings(req, server, operating_points)
            if cost < min_cost:
                min_cost = cost
                chosen_server = server

        assert chosen_server is not None and chosen_server in servers, f"Chosen server {chosen_server} is invalid."

        await PREFILL_LOCK.acquire()
        rank = int(re.search(r"rank-(\d+)", req.adapter_dir).group(1))
        if rank not in REMAINING_PREFILLS_PER_SERVER[chosen_server]:
            REMAINING_PREFILLS_PER_SERVER[chosen_server][rank] = 0
        REMAINING_PREFILLS_PER_SERVER[chosen_server][rank] += req.prompt_len
        PREFILL_LOCK.release()
            
        await DECODE_LOCK.acquire()
        rank = int(re.search(r"rank-(\d+)", req.adapter_dir).group(1))
        if rank not in REMAINING_DECODES_PER_SERVER[chosen_server]:
            REMAINING_DECODES_PER_SERVER[chosen_server][rank] = 0
        REMAINING_DECODES_PER_SERVER[chosen_server][rank] += (req.output_len - 1)
        DECODE_LOCK.release()
        
        task = asyncio.create_task(
            send_request(
                backend,
                chosen_server,
                req.req_id,
                req.model_dir,
                req.adapter_dir,
                req.prompt,
                req.prompt_len,
                req.output_len,
                output,
                debug,
                arrival_time
            )
        )
        tasks.append(task)
    latency = await asyncio.gather(*tasks)
    return latency

# def ema_next(values: list, alpha: float = 0.5):
#     assert values, "no values found when computing ema"
#     ema = values[-1]
#     for x in values[-2::-1]:
#         ema = alpha * x + (1 - alpha) * ema
#     return ema

def ema_next(values: list, alpha: float = 0.5):
    
    assert values, "no values found when computing ema"
    model = LinearRegression()
    model.fit(np.arange(len(values)).reshape(-1, 1), values)

    # Predict a value beyond known points
    extrapolated_value = model.predict(np.array([[len(values)]])).item()
    return max(min(values), extrapolated_value, 1)

def compare_with_prev_alloc(
    adapter_groups,
    prev_adapter_groups,
    assigned_instances_per_rank,
    prev_assigned_instances_per_rank,
    adapter_to_tps=None,
    jaccard_threshold=0.5,
):
    """
    Compares the current and previous allocations to identify any changes in server assignments.
    assigned_instances_per_rank: {rank: [server_indices]}
    adapter_groups: [(adapter_name, routing_probability),]
    """
    server_rename_map = {}  # {old_server: new_server}

    for rank, prev_servers in prev_assigned_instances_per_rank.items():
        if rank in assigned_instances_per_rank:
            curr_servers = assigned_instances_per_rank[rank]
            if len(prev_servers) == len(curr_servers):
                for old_server, new_server in zip(prev_servers, curr_servers):
                    server_rename_map[old_server] = new_server
            else:
                for prev_server in prev_servers:
                    old_adapters = set(
                        adapter for adapter, _ in prev_adapter_groups[prev_server]
                    )
                    for curr_server in curr_servers:
                        new_adapters = set(
                            adapter for adapter, _ in adapter_groups[curr_server]
                        )
                        intersection_adapters_sum_tps = sum(
                            adapter_to_tps.get(adapter, 0)
                            for adapter in old_adapters & new_adapters
                        )
                        union_adapters_sum_tps = sum(
                            adapter_to_tps.get(adapter, 0)
                            for adapter in old_adapters | new_adapters
                        )
                        if (
                            union_adapters_sum_tps > 0
                            and intersection_adapters_sum_tps / union_adapters_sum_tps
                            > jaccard_threshold
                        ):
                            server_rename_map[curr_server] = prev_server
                            break

    return server_rename_map


def ensure_all_placed(adapter_groups, epsilon: float = 0.1):
    """
    Ensure that the probability of adapter placements sums to within epsilon of 1 for each adapter
    """

    utils = {}
    for group in adapter_groups:
        for adapter, util in group:
            if adapter in utils:
                utils[adapter] += util
            else:
                utils[adapter] = util

    for adapter, util in utils.items():
        assert (
            abs(util - 1) < epsilon
        ), f"Adapter {adapter} not fully placed, util={util}"


def select_server(
    server_map, probability_sum, req, adapter_groups=None, adapter_demand=None
):
    """
    Selects a server for the given request based on the provided probability distribution.
    """
    available_servers, prob_thresholds = (
        server_map[req.adapter_dir],
        probability_sum[req.adapter_dir],
    )
    rand_prob = random.random()
    try:
        chosen_server = available_servers[min(
            bisect.bisect_left(prob_thresholds, rand_prob),
            len(available_servers) - 1
        )]
    except Exception as e:
        print(
            f"Error in bisecting {prob_thresholds} with rand_prob {rand_prob}, available_servers {available_servers}: {e}. Falling back to first in list if exists."
        )
        print(server_map)
        print(probability_sum)
        print(adapter_groups)
        print(adapter_demand)
        if available_servers:
            chosen_server = available_servers[0]
        else:
            raise Exception(f"No available servers for adapter {req.adapter_dir}")
    return chosen_server


def flatten_dict_values(d):
    vals = list(d.values())
    if not vals:
        return []

    if all(isinstance(v, str) for v in vals):
        return vals

    if all(isinstance(v, list) for v in vals):
        return list(chain.from_iterable(vals))

    raise TypeError(
        "Dictionary values are not uniform: mixture of str and list detected."
    )


async def benchmark_system(
    backend: str,
    server_map: str,
    adapter_dirs,
    input_requests: List[Tuple[str, str, str, int, int]],
    output,
    debug=False,
) -> None:
    start = time.time()
    step_idx = 0
    debug = False
    prev_alloc = None
    prev_rank_assigned_instances = None
    tasks: List[asyncio.Task] = []
    last_time = input_requests[0]
    step = 60
    probability_sum = defaultdict(
        list
    )  # adapter -> [prob of server 1, prob of server 1 + prob of server 2, ...]
    ema_lookback_seconds = 300
    ema_alpha = 0.5
    adapter_window_history = defaultdict(
        lambda: deque()
    )  # adapter -> deque of (window end time, tps), with newest on right

    # --- Cold miss tracking ---
    # Build initial per-server adapter sets from the incoming server_map (adapter -> server)
    prev_server_to_adapters = defaultdict(set)  # server -> set of adapter names
    for adapter, srv in server_map.items():
        if isinstance(srv, list):
            for s in srv:
                prev_server_to_adapters[s].add(adapter)
        else:
            prev_server_to_adapters[srv].add(adapter)
    # Cold miss tracking starts after first reallocation (step >= 1); initial adapters are not "new"
    cold_miss_remaining = defaultdict(set)  # server -> set of adapters not yet hit
    # Per (step_idx, server): (set_of_new_adapters, list_of_cold_miss_task_indices, total_req_count)
    cold_miss_step_data = {}  # (step_idx, server) -> [set(new_adapters), [task_indices], [arrival_times], req_count]
    current_cold_step = 0  # tracks which step current dispatches belong to
    cold_step_time_ranges = {0: [0.0, None]}  # step_idx -> [start_time, end_time] in request-time seconds
    # --- End cold miss tracking init ---

    for req in input_requests:
        arrival_time = start + req.req_time
        if req.req_time > last_time.req_time // 1 + step:
            alloc_start = time.time()
            demand_tps = {a: 0 for a in adapter_dirs}
            index = input_requests.index(last_time)
            window_end_time = req.req_time
            while (
                index < len(input_requests)
                and input_requests[index].req_time < req.req_time
            ):
                # rank = int(re.search(r'rank-(\d+)', input_requests[index].adapter_dir).group(1))
                r = input_requests[index]
                demand_tps[input_requests[index].adapter_dir] = (
                    demand_tps.get(r.adapter_dir) + (r.prompt_len + r.output_len) / step
                )
                index += 1
            
            for adapter, raw_tps in demand_tps.items():
                history = adapter_window_history[adapter]
                history.append((window_end_time, raw_tps))
                
                history_cutoff_time = window_end_time - ema_lookback_seconds
                while history and history[0][0] < history_cutoff_time:
                    history.popleft()
                    
                tps_values = [tps for _, tps in history]
                demand_tps[adapter] = ema_next(tps_values, alpha=ema_alpha)
            
            adapter_demand = []
            adapter_name_to_tps = {}
            for adapter, tps in demand_tps.items():
                rank = int(re.search(r"rank-(\d+)", adapter).group(1))
                adapter_demand.append((rank, tps, adapter))  # tps here is expected tps
                adapter_name_to_tps[adapter] = tps
            adapter_demand.sort(reverse=True)

            rank_wise_demand = {}
            for rank, tps, adapter in adapter_demand:
                if rank not in rank_wise_demand:
                    rank_wise_demand[rank] = 0
                rank_wise_demand[rank] += tps

            # server_tps = {
            #     8: 2400,
            #     16: 2100,
            #     32: 1900,
            #     64: 1700,
            #     128: 1600,
            # }  # operating point, fn of max rank, old NC24ads-hipri 8xA100 40GB
            # server_tps = {
            #     8: 2725,
            #     16: 2700,
            #     32: 2675,
            #     64: 2625,
            #     128: 2525,
            # }  # operating point, fn of max rank, 4xA100 80GB
            # server_tps = {
            #     8: 5500,
            #     16: 5400,
            #     32: 5250,
            #     64: 5000,
            #     128: 4500,
            # }  # operating point, fn of max rank, ND96asrv 8xA100 80GB
            server_tps = {8: 5950, 16: 5150, 32: 4700, 64: 5050, 128: 4650} # prod 7b tp1, NC24ads

            rank_instance_demand = {}
            for rank, tps in rank_wise_demand.items():
                rank_instance_demand[rank] = tps / server_tps[rank]

            total_instance_demand = sum(rank_instance_demand.values())

            flattened_server_map = flatten_dict_values(server_map)
            servers = sorted(list(set(flattened_server_map)))
            adapter_groups = [[] for _ in servers]
            num_servers = len(servers)
            server_occupied_tps = [0] * num_servers
            server_max_rank = [0] * num_servers

            target_util = total_instance_demand / len(servers)
            # assert (
            #     target_util <= 1
            # ), f"Target utilization exceeds 1, need more servers: {target_util}"

            with open("allocation_log.txt", "a") as f:
                f.write("\n\n************************************")
                f.write(
                    f"step {step_idx} @ time {last_time.req_time} to {req.req_time}:"
                )

            # * checking compatibility
            # TODO: this was the old tps based compatibility, can we remove?
            # TODO: since we check compatibility with the rank instance budget now
            rank_instance_budget = [
                (
                    rank,
                    sum(tps for r, tps, _ in adapter_demand if r == rank)
                    / rank_max_tps,
                )
                for rank, rank_max_tps in server_tps.items()
            ]
            sorted_budgets = sorted(
                rank_instance_budget, key=lambda x: x[1], reverse=True
            )
            # assert (
            #     sum(budget for _, budget in rank_instance_budget) <= num_servers
            # ), "Exceeded server budget"

            # * rounding
            rounded_budgets = [
                (budget, rank, round(budget / target_util))
                for rank, budget in sorted_budgets
                if round(budget / target_util) > 0
            ]
            if len(rounded_budgets) == 0:
                # Force at least one instance for the largest rank budget
                largest_rank, largest_budget = max(sorted_budgets, key=lambda x: x[1])
                rounded_budgets = [(largest_budget, largest_rank, 1)]
                
            rounded_budgets.sort(reverse=True, key=lambda x: (x[0] / x[2], x[1]))
            zero_budgets = [
                (budget, rank, round(budget / target_util))
                for rank, budget in sorted_budgets
                if round(budget / target_util) == 0
            ]
            sum_rounded_off_budgets = sum(budget for _, _, budget in rounded_budgets)
            while sum_rounded_off_budgets < num_servers:
                # print("Increasing instances")
                first = rounded_budgets[0]
                rounded_budgets = [
                    (first[0], first[1], first[2] + 1)
                ] + rounded_budgets[1:].copy()
                rounded_budgets.sort(reverse=True, key=lambda x: (x[0] / x[2], x[1]))
                sum_rounded_off_budgets = sum(
                    budget for _, _, budget in rounded_budgets
                )
            rounded_budgets.sort(key=lambda x: x[1])
            sum_rounded_off_budgets = sum(budget for _, _, budget in rounded_budgets)
            while sum_rounded_off_budgets > num_servers:
                # print("Decreasing instances")
                first = rounded_budgets[0]
                assert first[2] > 0, "Cannot reduce instances further"
                rounded_budgets = [
                    (first[0], first[1], first[2] - 1)
                ] + rounded_budgets[1:].copy()
                rounded_budgets.sort(key=lambda x: x[1])
                sum_rounded_off_budgets = sum(
                    budget for _, _, budget in rounded_budgets
                )

            # * balanced allocation within assigned instances
            ranks_with_assigned_instances = [(x[1], x[2]) for x in rounded_budgets]
            rank_assigned_instances = {}
            ranks_with_zero_instances = [x[1] for x in zero_budgets]
            last_used_server = 0
            server_util = [0] * num_servers
            leftovers = []
            for rank in ranks_with_zero_instances:
                leftovers.extend(
                    [
                        (idx, adapter, 1.0)
                        for idx, adapter in enumerate(adapter_demand)
                        if adapter[0] == rank
                    ]
                )
            for rank, budget in ranks_with_assigned_instances:
                # assign adapters of rank to num_instances greedily
                # maybe sort in descending tps order and assign fractionally from the left

                # get all adapters of this rank
                adapters_of_rank = [
                    (idx, adapter)
                    for idx, adapter in enumerate(adapter_demand)
                    if adapter[0] == rank
                ]
                adapters_of_rank.sort(
                    reverse=True, key=lambda x: x[1][1]
                )  # sort by tps descending
                servers_used = 0
                if budget > 0:
                    rank_assigned_instances[rank] = list(
                        range(last_used_server, last_used_server + budget)
                    )
                for adapter_idx, adapter in adapters_of_rank:
                    tps, adapter_name = adapter[1], adapter[2]
                    expected_util = tps / server_tps[rank]
                    _expected_util = expected_util
                    while expected_util > 1e-4 and servers_used < budget:
                        # assign as much as possible to this server
                        if servers_used >= budget:
                            raise Exception(
                                f"Ran out of servers: used {servers_used}, budget {budget} at adapter {adapter} at index {adapter_idx}"
                            )
                        server_idx = last_used_server + servers_used
                        # assign to this server
                        max_addable_util = min(
                            target_util - server_util[server_idx], expected_util
                        )
                        adapter_groups[server_idx].append(
                            [adapter_name, max_addable_util / _expected_util]
                        )
                        server_occupied_tps[server_idx] += (
                            max_addable_util * server_tps[rank]
                        )

                        # adapters_placed[adapter_idx] = True
                        server_max_rank[server_idx] = max(
                            server_max_rank[server_idx], rank
                        )
                        server_util[server_idx] += max_addable_util
                        if server_util[server_idx] >= target_util:
                            servers_used += 1
                        expected_util -= max_addable_util
                    if expected_util > 1e-4:
                        leftovers.append(
                            (adapter_idx, adapter, expected_util / _expected_util)
                        )
                last_used_server += budget

            # * leftovers
            space_left = (target_util * num_servers) - sum(server_util)
            demand_left = 0
            for _, adapter_tuple, util_fraction in leftovers:
                demand_left += (adapter_tuple[1] * util_fraction) / server_tps[
                    adapter_tuple[0]
                ]
            # Guard: if rounding earlier caused an impossible leftover (slight FP drift or forced instance), scale leftovers proportionally.
            rescaled_leftovers = False
            if demand_left - space_left >= 1e-3:
                with open("allocation_log.txt", "a") as f:
                    f.write(
                        f"[warn] Leftover demand ({demand_left:.6f}) exceeds space ({space_left:.6f}). Scaling leftovers proportionally.\n"
                    )
                if demand_left > 0 and space_left > 0:
                    rescaled_leftovers = True
                    scale = space_left / demand_left
                    new_leftovers = []
                    for adapter_idx, adapter_tuple, util_fraction in leftovers:
                        new_leftovers.append(
                            (adapter_idx, adapter_tuple, util_fraction * scale)
                        )
                    leftovers = new_leftovers
                else:
                    # No capacity left; drop leftover demands (will be completed in next window)
                    leftovers = []
                    with open("allocation_log.txt", "a") as f:
                        f.write(
                            "[warn] Dropping leftovers due to zero capacity; will retry next window.\n"
                        )

            leftovers.sort(
                reverse=True, key=lambda x: (x[1][1])
            )  # sort by tps descending
            round_robin_server_idx = 0
            for _, adapter_tuple, util_fraction in leftovers:
                adapter_rank, adapter_demand_tps, adapter_name = adapter_tuple
                _expected_util = adapter_demand_tps / server_tps[adapter_rank]
                adapter_demand_tps *= util_fraction
                expected_util = adapter_demand_tps / server_tps[adapter_rank]
                # go through the servers with higher max rank and fit the fractional demand until target utilization
                server_idx = 0
                allocated_adapter = False

                if expected_util < 1e-3:
                    adapter_groups[round_robin_server_idx].append([adapter_name, util_fraction])
                    server_max_rank[round_robin_server_idx] = max(
                        server_max_rank[round_robin_server_idx], adapter_rank
                    )
                    server_occupied_tps[round_robin_server_idx] += (
                        util_fraction * server_tps[adapter_rank]
                    )
                    server_util[round_robin_server_idx] += util_fraction
                    allocated_adapter = True
                    round_robin_server_idx = (round_robin_server_idx + 1) % num_servers
                    continue

                # while (
                #     expected_util > 1e-3
                #     and server_idx < num_servers
                #     and not allocated_adapter
                # ):
                #     if (
                #         server_max_rank[server_idx] >= adapter_rank
                #         and server_util[server_idx] < target_util
                #     ):
                #         max_addable_util = min(
                #             target_util - server_util[server_idx], expected_util
                #         )
                #         adapter_groups[server_idx].append(
                #             [adapter_name, max_addable_util / _expected_util]
                #         )
                #         server_occupied_tps[server_idx] += (
                #             max_addable_util * server_tps[adapter_rank]
                #         )
                #         server_max_rank[server_idx] = max(
                #             server_max_rank[server_idx], adapter_rank
                #         )
                #         server_util[server_idx] += max_addable_util
                #         expected_util -= max_addable_util
                #     server_idx += 1

                server_available = True
                while expected_util > 1e-3 and server_available and not allocated_adapter:
                    chosen_server = -1
                    max_addable_util = -np.inf
                    for server_idx in range(num_servers):
                        if (
                            server_max_rank[server_idx] >= adapter_rank
                            and server_util[server_idx] < target_util
                        ):
                            possible_addable_util = min(
                                target_util - server_util[server_idx], expected_util
                            )
                            if possible_addable_util > max_addable_util:
                                max_addable_util = possible_addable_util
                                chosen_server = server_idx
                    if chosen_server >= 0:
                        adapter_groups[chosen_server].append(
                            [adapter_name, max_addable_util / _expected_util]
                        )
                        server_occupied_tps[chosen_server] += (
                            max_addable_util * server_tps[adapter_rank]
                        )
                        server_max_rank[chosen_server] = max(
                            server_max_rank[chosen_server], adapter_rank
                        )
                        server_util[chosen_server] += max_addable_util
                        expected_util -= max_addable_util
                    else:
                        server_available = False
                        
                    if expected_util < 1e-3:
                        allocated_adapter = True
                    
                
                if expected_util == 0:
                    allocated_adapter = True

                if not allocated_adapter:
                    server_idx = 0
                    # we could not find a server with rank >= this adapters rank
                    # need to colocate with a lower rank
                    while expected_util > 1e-3 and server_idx < num_servers:
                        if server_util[server_idx] < target_util:
                            max_addable_util = min(
                                target_util - server_util[server_idx], expected_util
                            )
                            adapter_groups[server_idx].append(
                                [adapter_name, max_addable_util / _expected_util]
                            )
                            server_occupied_tps[server_idx] += (
                                max_addable_util * server_tps[adapter_rank]
                            )
                            server_max_rank[server_idx] = max(
                                server_max_rank[server_idx], adapter_rank
                            )
                            server_util[server_idx] += max_addable_util
                            expected_util -= max_addable_util
                        server_idx += 1

                if expected_util > 1e-3:
                    try:
                        for i in range(num_servers):
                            if server_util[i] + expected_util <= 1:
                                adapter_groups[i].append(
                                    [adapter_name, expected_util / _expected_util]
                                )
                                server_occupied_tps[i] += (
                                    expected_util * server_tps[adapter_rank]
                                )
                                server_max_rank[i] = max(
                                    server_max_rank[i], adapter_rank
                                )
                                server_util[i] += expected_util
                                expected_util = 0
                                break
                    except Exception as e:
                        raise Exception(
                            f"Could not allocate adapter {adapter_name} with rank {adapter_rank} and tps {adapter_demand_tps}, leftover util {expected_util}"
                        )

            print(adapter_groups)
            with open("allocation_log.txt", "a") as f:
                f.write(f"\n{last_time} Adapter groups:\n")
                for i, group in enumerate(adapter_groups):
                    f.write(
                        f"  Server {servers[i]}: {[(adapter, used_util, adapter_name_to_tps[adapter] * used_util) for adapter, used_util in group]}\n"
                    )
                    f.write(
                        f"  Server {servers[i]} total tps: {server_occupied_tps[i]}\n"
                    )
                    f.write(
                        f"  Server {servers[i]} max tps: {server_tps.get(server_max_rank[i], 0)}\n"
                    )
                    f.write(f"  Server {servers[i]} util: {server_util[i]}\n")
                    f.write(f"  Server {servers[i]} max rank: {server_max_rank[i]}\n")
                f.write("************************************\n\n")

            # if not rescaled_leftovers:
                # ensure_all_placed(adapter_groups)
            print(
                f"All adapters placed successfully for step {step_idx} from time {last_time.req_time} to {req.req_time}"
            )

            # * compare with last iteration
            server_rename_map = None
            if prev_alloc is not None and prev_rank_assigned_instances is not None:
                adapter_to_tps = {adapter: tps for _, tps, adapter in adapter_demand}
                server_rename_map = compare_with_prev_alloc(
                    adapter_groups=adapter_groups,
                    prev_adapter_groups=prev_alloc,
                    assigned_instances_per_rank=rank_assigned_instances,
                    prev_assigned_instances_per_rank=prev_rank_assigned_instances,
                    adapter_to_tps=adapter_to_tps,
                )
                with open("allocation_log.txt", "a") as f:
                    f.write(
                        f"Prev rank assigned instances (step {step_idx - 1}): {prev_rank_assigned_instances}\n"
                    )
                    f.write(
                        f"Curr rank assigned instances (step {step_idx}): {rank_assigned_instances}\n"
                    )
                if server_rename_map:
                    print("Server renames detected:", server_rename_map)
                    with open("allocation_log.txt", "a") as f:
                        f.write(f"Server renames detected: {server_rename_map}\n")
                    if debug:
                        print(
                            f"Prev rank assigned instances (step {step_idx - 1}):",
                            prev_rank_assigned_instances,
                        )
                        print(
                            f"Curr rank assigned instances (step {step_idx}):",
                            rank_assigned_instances,
                        )

            server_map = defaultdict(list)  # adapter -> [server1, server2, ...]
            probability_sum = defaultdict(
                list
            )  # adapter -> [prob of server 1, prob of server 1 + prob of server 2, ...]
            for i, server in enumerate(servers):
                for adapter, util in adapter_groups[i]:
                    if not server_rename_map or server not in server_rename_map.keys():
                        server_map[adapter].append(server)
                        probability_sum[adapter].append(
                            probability_sum[adapter][-1] + util
                            if probability_sum[adapter]
                            else util
                        )
                    else:
                        server_map[adapter].append(server_rename_map[server])

            # --- Cold miss: compute new adapters for the new allocation ---
            current_server_to_adapters = defaultdict(set)
            for i_srv, server_url in enumerate(servers):
                for adapter_name_ag, _ in adapter_groups[i_srv]:
                    current_server_to_adapters[server_url].add(adapter_name_ag)
            # Determine newly placed adapters per server
            cold_miss_remaining = defaultdict(set)
            for srv_url in set(list(current_server_to_adapters.keys()) + list(prev_server_to_adapters.keys())):
                new_adapters_on_srv = current_server_to_adapters.get(srv_url, set()) - prev_server_to_adapters.get(srv_url, set())
                cold_miss_remaining[srv_url] = set(new_adapters_on_srv)
            prev_server_to_adapters = current_server_to_adapters
            # --- End cold miss new-adapter computation ---

            prev_alloc = adapter_groups.copy()
            prev_rank_assigned_instances = rank_assigned_instances.copy()
            # Close previous step's time range and open a new one
            cold_step_time_ranges[current_cold_step][1] = req.req_time
            step_idx += 1
            current_cold_step = step_idx  # set AFTER increment so post-reallocation requests get a new step
            cold_step_time_ranges[current_cold_step] = [req.req_time, None]
            last_time = req
            
            alloc_end = time.time()
            with open("allocation_log.txt", "a") as f:
                f.write(
                    f"Allocation computation time: {alloc_end - alloc_start:.4f} s\n"
                )
            
            with open("server_map.json", "a") as f:
                f.write("\n")
                json.dump(server_map, f, indent=4)
                
            with open("demand_tps.json", "a") as f:
                f.write("\n")
                json.dump(adapter_name_to_tps, f, indent=4)
        sleep_time = arrival_time - time.time()
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        if debug:
            print(
                f"{req.req_id} {req.req_time:.5f} wait {start + req.req_time - time.time():.5f} "
                f"{req.adapter_dir}"
            )
        # print(req)

        # * sample from the adapter groups based on the placement probabilities
        chosen_server = None
        if probability_sum:
            # print("Sampling server based on placement probabilities", flush=True)
            chosen_server = select_server(
                server_map, probability_sum, req, adapter_groups, adapter_demand
            )
        else:
            # print("Choosing first server always", flush=True)
            if server_map[req.adapter_dir]:
                if type(server_map[req.adapter_dir]) is list:
                    chosen_server = server_map[req.adapter_dir][0]
                else:
                    chosen_server = server_map[req.adapter_dir]

        # print(f"Request {req.req_id} for adapter {req.adapter_dir} assigned to server {chosen_server} at time {req.req_time}", flush=True)

        # --- Cold miss detection (only after first reallocation, step >= 1) ---
        is_cold_miss = False
        if current_cold_step >= 1 and chosen_server is not None and req.adapter_dir in cold_miss_remaining.get(chosen_server, set()):
            is_cold_miss = True
            cold_miss_remaining[chosen_server].discard(req.adapter_dir)
        if current_cold_step >= 1:
            task_index = len(tasks)
            key = (current_cold_step, chosen_server)
            if key not in cold_miss_step_data:
                cold_miss_step_data[key] = [set(), [], [], 0]  # [cold_miss_adapter_names, task_indices, arrival_times, req_count]
            cold_miss_step_data[key][3] += 1  # total request count
            if is_cold_miss:
                cold_miss_step_data[key][0].add(req.adapter_dir)  # record which adapter had the cold miss
                cold_miss_step_data[key][1].append(task_index)  # store task index for TTFT lookup
                cold_miss_step_data[key][2].append(round(req.req_time, 4))  # arrival time
        # --- End cold miss detection ---

        task = asyncio.create_task(
            send_request(
                backend,
                chosen_server,
                req.req_id,
                req.model_dir,
                req.adapter_dir,
                req.prompt,
                req.prompt_len,
                req.output_len,
                output,
                debug,
                arrival_time
            )
        )
        tasks.append(task)
    latency = await asyncio.gather(*tasks)

    # --- Cold miss: write CSV ---
    # Close the final step's time range
    if input_requests:
        cold_step_time_ranges[current_cold_step][1] = input_requests[-1].req_time
    with open("cold_misses.csv", "w") as cm_f:
        cm_f.write("step_idx,step_time_range,server,new_adapters,num_cold_miss_req,perc_cold_miss_req,cold_miss_req_ttfts,cold_miss_req_arrival_times\n")
        for (s_idx, srv), (cold_miss_adapters, cold_task_indices, cold_arrival_times, total_reqs) in sorted(cold_miss_step_data.items()):
            t_start, t_end = cold_step_time_ranges.get(s_idx, [0.0, 0.0])
            time_range_str = f"{t_start:.0f}-{t_end:.0f}s"
            new_adapters_list = sorted(cold_miss_adapters)  # only adapters that actually had cold miss requests
            num_cold = len(cold_task_indices)
            perc_cold = (num_cold / total_reqs * 100.0) if total_reqs > 0 else 0.0
            cold_ttfts = []
            for t_idx in cold_task_indices:
                result = latency[t_idx]
                if result is not None and len(result) > 3 and result[3] is not None:
                    cold_ttfts.append(round(result[3], 6))
            # Format lists as semicolon-separated quoted strings for CSV
            new_adapters_str = '"' + ';'.join(new_adapters_list) + '"'
            ttfts_str = '"' + ';'.join(str(t) for t in cold_ttfts) + '"'
            arrival_times_str = '"' + ';'.join(str(t) for t in cold_arrival_times) + '"'
            cm_f.write(f"{s_idx},{time_range_str},{srv},{new_adapters_str},{num_cold},{perc_cold:.4f},{ttfts_str},{arrival_times_str}\n")
    print(f"Cold miss data written to cold_misses.csv ({len(cold_miss_step_data)} rows)")
    # --- End cold miss CSV ---

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


def get_res_stats(
    per_req_latency, benchmark_time, backend, warmup_time=0, warmup_num=0
):
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

    strip_throughput = (len(per_req_latency) - warmup_num) / max(
        1e-9, benchmark_time - warmup_time
    )
    print(f"Throughput strip: {strip_throughput:.2f} requests/s")

    # Exclude first warmup_num requests from further statistics
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    if warmup_num >= len(per_req_latency):
        raise ValueError(
            "Warmup number exceeds number of requests whose latency was logged"
        )
    else:
        stats_base = per_req_latency[warmup_num:]

    # each entry in stats_base: (prompt_len, output_len, request_latency, first_token_latency, tbt, scheduling_delay)
    e2e_latencies = [latency for _, _, latency, _, _, _ in stats_base]
    per_token_latencies = [
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency, _, _, _ in stats_base
        if (prompt_len + output_len) > 0
    ]
    per_output_token_latencies = [
        latency / output_len
        for _, output_len, latency, _, _, _ in stats_base
        if output_len > 0
    ]
    first_token_latencies = [latency for _, _, _, latency, _, _ in stats_base]
    tbts = [latency for _, _, _, _, latency, _ in stats_base]
    num_abort = len([i for i in stats_base if i[3] is None])
    abort_satisfaction = [0] * num_abort
    satisfactions = [
        reward(latency) for _, _, _, latency, _, _ in stats_base
    ] + abort_satisfaction
    attainments = [
        attainment_func(latency) for _, _, _, latency, _, _ in stats_base
    ] + abort_satisfaction

    scheduling_delays = [delay[5] for delay in stats_base if len(delay) > 5]
    
    if len(scheduling_delays) == 0:
        scheduling_delays = [0.0] * len(stats_base)

    metrics = {
        "e2e": e2e_latencies,
        "per_token": per_token_latencies,
        "per_output_token": per_output_token_latencies,
        "scheduling": scheduling_delays,
        "first_token": first_token_latencies,
        "tbt": tbts,
        "satisfaction": satisfactions,
        "attainment": attainments,
    }

    stats = {}
    for name, values in metrics.items():
        stats[name] = {
            "avg": np.mean(values),
            **{f"p{p}": np.percentile(values, p) for p in percentiles},
        }

    # dump results
    if backend == "dm":
        # TODO
        # single_gpu_peak_mem = peak_mem
        single_gpu_peak_mem = 0
    else:
        single_gpu_peak_mem = 0

    result = {
        "total_time": benchmark_time,
        "gpu_peak_mem": single_gpu_peak_mem,
        "num_abort": num_abort,
        "throughput": throughput,
        "strip_throughput": strip_throughput,
        "stats": stats,
        "backend": backend,
    }
    res = {"result": result}

    return res


def read_requests(trace_file):
    requests = []
    adapter_dirs = set()
    with open(trace_file, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            elements = line.split(",")
            requests.append(
                Request(
                    req_id=int(elements[0]),
                    model_dir=elements[1],
                    adapter_dir=elements[2],
                    prompt=dummy_prompt(int(elements[3])),
                    prompt_len=int(elements[3]),
                    output_len=int(elements[4]),
                    req_time=float(elements[5]),
                )
            )
            # requests.append((int(elements[0]),elements[1],elements[2],int(elements[3]),int(elements[4]),float(elements[5])))
            adapter_dirs.add(elements[2])
    requests.sort(key=lambda r: r.req_time)
    return list(adapter_dirs), requests


def run_exp(
    backend,
    servers,
    trace_file,
    output,
    debug=False,
    warmup_time: int = 60,
    warmup_num: int = 600,
):
    # first generate your data using real_trace/clean_chat_data.py
    # base_model = BASE_MODEL[model_setting]
    # adapter_dirs = LORA_DIR[model_setting]
    adapter_dirs, requests = read_requests(trace_file=trace_file)
    # print(requests)
    avg_prompt_len = np.mean([req.prompt_len for req in requests])
    avg_output_len = np.mean([req.output_len for req in requests])
    avg_len = np.mean([req.prompt_len + req.output_len for req in requests])
    print(
        "num_adapters",
        len(adapter_dirs),
        "num_requests",
        len(requests),
        "avg_len:",
        avg_len,
        "avg_prompt_len:",
        avg_prompt_len,
        "avg_output_len:",
        avg_output_len,
    )

    if debug:
        print("num requests:", len(requests))
        for req in requests[:4]:
            print(req)
    if backend == "baseline" or backend == "toppings":
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
        
        adapter_allocations = defaultdict(list)
        for adapter, server in server_map.items():
            adapter_allocations[server].append(adapter)
            
        # with open("server_map.csv", "w") as f:
        #     f.write("server,adapters\n")
        #     for server, adapters in adapter_allocations.items():
        #         f.write(f"{server},{' '.join(adapters)}\n")
                
        with open("server_map.json", "w") as f:
            json.dump(server_map, f, indent=4)
        
        if backend == "baseline":
            benchmark_start_time = time.time()
            per_req_latency = asyncio.run(
                benchmark_baseline(backend, server_map, requests, output, debug)
            )
            benchmark_end_time = time.time()
            benchmark_time = benchmark_end_time - benchmark_start_time
        elif backend == "toppings":
            benchmark_start_time = time.time()
            per_req_latency = asyncio.run(
                benchmark_toppings(backend, requests, output, servers, debug)
            )
            benchmark_end_time = time.time()
            benchmark_time = benchmark_end_time - benchmark_start_time
    elif backend == "system" or backend == "contiguous":
        # benchmark
        adapters = []
        for adapter in adapter_dirs:
            rank = int(re.search(r"rank-(\d+)", adapter).group(1))
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
        
        with open("server_map.json", "w") as f:
            json.dump(server_map, f, indent=4)
            
        if backend == "system":
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
            benchmark_start_time = time.time()
            per_req_latency = asyncio.run(
                benchmark_system(
                    backend, server_map, adapter_dirs, requests, output, debug
                )
            )
            benchmark_end_time = time.time()
        elif backend == "contiguous":
            benchmark_start_time = time.time()
            per_req_latency = asyncio.run(
                benchmark_baseline(backend, server_map, requests, output, debug)
            )
            benchmark_end_time = time.time()
        benchmark_time = benchmark_end_time - benchmark_start_time

    res = get_res_stats(
        per_req_latency,
        benchmark_time,
        backend,
        warmup_time=warmup_time,
        warmup_num=warmup_num,
    )

    with open(output, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["system", "baseline", "contiguous", "toppings"],
    )

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
    parser.add_argument(
        "--warmup-time",
        type=int,
        default=60,
        help="Number of seconds considered as warmup, excluded from stats",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=600,
        help="Number of requests considered as warmup, excluded from stats (rps * warmup-time)",
    )
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
        os.system(f"rm fine_{args.output}")
        results = []
    else:
        with open(args.output, "r") as f:
            lines = f.readlines()
        results = [json.loads(line)["config"] for line in lines]

    # for config in tqdm(suites, desc="suites"):
    #     if to_dict(config) not in results:
    stats = run_exp(
        args.backend,
        args.servers,
        args.trace_file_path,
        args.output,
        args.debug,
        args.warmup_time,
        args.warmup_requests,
    )
