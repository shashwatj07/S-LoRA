import pandas as pd
import numpy as np
import re
import glob
import json
import os
import random
import argparse
import bisect

from collections import defaultdict
from itertools import chain
from typing import List, Tuple

from trace import Request, dummy_prompt


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


def load_server_maps(server_maps_dir: str):
    """
    Load precomputed server maps of form server_map_step_{i}.json
    Returns:
      server_maps_by_step: {step_idx: {adapter: [server1, server2, ...]}}
      server_adapters_by_step: {step_idx: {server: set(adapters)}}
      max_step_idx: largest step index loaded
    """
    files = sorted(
        glob.glob(os.path.join(server_maps_dir, "server_map_step_*.json")),
        key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]),
    )
    server_maps_by_step = {}
    server_adapters_by_step = {}
    for fp in files:
        step_idx = int(os.path.basename(fp).split("_")[-1].split(".")[0])
        with open(fp, "r") as f:
            m = json.load(f)  # adapter -> [servers...]
        server_maps_by_step[step_idx] = m
        inv = defaultdict(set)
        for adapter, servers in m.items():
            # servers may be a str or list depending on generation
            if isinstance(servers, str):
                inv[servers].add(adapter)
            else:
                for s in servers:
                    inv[s].add(adapter)
        server_adapters_by_step[step_idx] = inv
    if not server_maps_by_step:
        raise ValueError(f"No server_map_step_*.json files found in {server_maps_dir}")
    max_step_idx = max(server_maps_by_step.keys())
    return server_maps_by_step, server_adapters_by_step, max_step_idx


def load_probability_sums(server_maps_dir: str):
    """
    Load precomputed probability sums of form probability_sum_step_{i}.json
    Returns:
      probability_sums_by_step: {step_idx: {adapter: [prob_sum1, prob_sum2, ...]}}
    """
    files = sorted(
        glob.glob(os.path.join(server_maps_dir, "probability_sum_step_*.json")),
        key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]),
    )
    probability_sums_by_step = {}
    for fp in files:
        step_idx = int(os.path.basename(fp).split("_")[-1].split(".")[0])
        with open(fp, "r") as f:
            m = json.load(f)  # adapter -> [prob_sum...]
        probability_sums_by_step[step_idx] = m
    if not probability_sums_by_step:
        raise ValueError(
            f"No probability_sum_step_*.json files found in {server_maps_dir}"
        )
    return probability_sums_by_step


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
        chosen_server = available_servers[
            min(
                bisect.bisect_left(prob_thresholds, rand_prob),
                len(available_servers) - 1,
            )
        ]
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


def write_server_trace(
    backend,
    server_maps_by_step,
    probability_sums_by_step,
    max_step_idx,
    step_size,
    servers,
    input_requests: List[Tuple[str, str, str, int, int]],
    output,
):
    with open(output, "w") as f:
        f.write("req_id,model_dir,adapter_dir,prompt_len,output_len,req_time,server\n")
    for req in input_requests:
        step_idx = int(req.req_time // step_size)
        if step_idx > max_step_idx:
            step_idx = max_step_idx

        server_map_for_step = None
        if backend == "system":
            server_map_for_step = server_maps_by_step.get(step_idx)
        else:
            server_map_for_step = server_maps_by_step

        if server_map_for_step is None:
            raise ValueError(f"No server map found for step {step_idx}")

        adapter_servers = server_map_for_step.get(req.adapter_dir)
        if adapter_servers is None:
            raise ValueError(
                f"No servers found for adapter {req.adapter_dir} at step {step_idx}"
            )

        if isinstance(adapter_servers, str):
            chosen_server = adapter_servers
        else:
            if backend == "system":
                chosen_server = select_server(
                    server_map_for_step, probability_sums_by_step.get(step_idx, {}), req
                )
            else:
                raise ValueError(
                    f"Multiple servers found for adapter {req.adapter_dir} at step {step_idx} but backend is {backend}, expected 'system' backend for multiple server selection."
                )

        if chosen_server not in servers:
            continue

        with open(output, "a") as f:
            f.write(
                f"{req.req_id},{req.model_dir},{req.adapter_dir},{req.prompt_len},{req.output_len},{req.req_time},{chosen_server}\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["system", "baseline", "contiguous"],
    )
    parser.add_argument("--trace-file-path", required=True)

    parser.add_argument("--servers", "-s", type=str, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, required=True)

    parser.add_argument(
        "--server-maps-dir",
        type=str,
        default=None,
        required=True,
        help="Directory containing server maps for each time window as server_map_step_{i}.json and probability_sums_step_{i}.json if system",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=60,
        help="Time duration of each step/window in seconds",
    )
    args = parser.parse_args()

    os.system(f"rm {args.output}")

    adapter_dirs, requests = read_requests(trace_file=args.trace_file_path)
    if args.backend == "system":
        server_maps_by_step, _, max_step_idx = load_server_maps(args.server_maps_dir)
        probability_sums_by_step = load_probability_sums(args.server_maps_dir)
    else:
        server_maps_by_step = json.load(open(os.path.join(args.server_maps_dir, "server_map.json"), "r"))
        max_step_idx = np.inf
        probability_sums_by_step = None
        
    write_server_trace(
        backend=args.backend,
        server_maps_by_step=server_maps_by_step,
        probability_sums_by_step=probability_sums_by_step,
        max_step_idx=max_step_idx,
        step_size=args.step_size,
        servers=args.servers,
        input_requests=requests,
        output=args.output,
    )