import argparse
import random
import pandas as pd
import numpy as np
import os

# req_id=0, model_dir=huggyllama/llama-7b, adapter_dir=dummy-lora-7b-rank-32-0, prompt_len=500, output_len=128, req_time=3.010121430917521   
def parse_key_value(s):
    try:
        key, value = s.split("=")
        return int(key), int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid format: '{s}', expected format key=value")

def get_weights(size, dist):
    if dist == "pareto":
        alpha = 1.5
        x_m = 1

        weights = [(x_m / (i + 1)) ** alpha for i in range(size)]

        total_weight = sum(weights)
        dist = [w / total_weight for w in weights]
        random.shuffle(dist)
        return dist
    elif dist == "skew":
        return [1.0 for _ in range(20)] + [4.0 for _ in range(5)]
    elif dist == "uniform":
        return [1.0 / size for _ in range(size)]

def get_times(size, end_time, dist, burst_size, burst_interval):
    if dist == "uniform":
        return np.random.uniform(0, end_time, size)
    elif dist == "even":
        return np.linspace(0, end_time, size)
    elif dist == "poisson":
        #! the number of samples is not guaranteed to be rps * end_time
        lam = size / end_time  # average rate of events per unit time
        inter_arrival_times = np.random.exponential(1/lam, size)
        arrival_times = np.cumsum(inter_arrival_times)
        return arrival_times[arrival_times <= end_time]
    elif dist == "bursty":
        # times = []
        # current_time = 0
        # burst_size = int(size // (end_time / (burst_interval * 2)))
        # while len(times) < size:
        #     burst_times = np.random.uniform(current_time, current_time + burst_interval, burst_size)
        #     times.extend(burst_times)
        #     current_time += burst_interval * 2  # wait for the next burst
        # return np.array(sorted(times))[:size]
        if size <= 3:
            return np.sort(np.random.uniform(0, end_time, size))

        burst_fraction = random.uniform(0.65, 0.8)
        burst_reqs = max(1, int(size * burst_fraction))
        bg_reqs = size - burst_reqs

        num_bursts = min(max(1, burst_reqs), random.randint(end_time // 60, end_time // 20))
        min_frac, max_frac = 0.01, 0.05
        durations = np.random.uniform(min_frac * end_time, max_frac * end_time, num_bursts)

        weights = np.random.dirichlet([1.0] * num_bursts)
        burst_counts = np.maximum(1, np.round(weights * burst_reqs).astype(int))
        diff = burst_reqs - burst_counts.sum()
        if diff != 0:
            idx = np.argmax(burst_counts)
            burst_counts[idx] += diff

        starts = np.random.uniform(0, end_time, num_bursts)
        durations = np.minimum(durations, np.maximum(1e-6, end_time - 1e-6))

        burst_times_list = []
        for s, d, c in zip(starts, durations, burst_counts):
            if s + d > end_time:
                s = max(0.0, end_time - d)
            burst_times_list.append(np.random.uniform(s, s + d, c))

        burst_times = np.concatenate(burst_times_list) if burst_times_list else np.array([])

        if bg_reqs > 0:
            bg_times = np.random.uniform(0, end_time, bg_reqs)
            all_times = np.concatenate([burst_times, bg_times])
        else:
            all_times = burst_times

        if len(all_times) > size:
            all_times = all_times[:size]

        return np.sort(all_times)

def main():
    parser = argparse.ArgumentParser(description="Generate Traces")
    parser.add_argument("--output", "-o", type=str, default="./traces",
                        help="Path to write traces")
    parser.add_argument("--model", "-m", type=str, required=True, help="HF model mdentifier")
    parser.add_argument("--rank-frequency", "-r", type=parse_key_value, nargs="+", required=True,
                        help="Pass rank=frequency pairs like 8=3 16=4 128=6")
    parser.add_argument("--distribution", "-d", type=str, default="pareto", help="Popularity distribution of adapters")
    parser.add_argument("--arrival-pattern", "-a", type=str, default="uniform", help="Arrival pattern of requests")
    parser.add_argument("--burst-size", type=int, default=10, help="Burst size if bursty arrival pattern is chosen")
    parser.add_argument("--burst-interval", type=float, default=2, help="Burst interval if bursty arrival pattern is chosen")
    parser.add_argument("--rps", type=float, help="Total requests per second")
    parser.add_argument("--time", "-t", type=int, default=5*60, help="Total time duration")
    
    args = parser.parse_args()
    ranks_dict = dict(args.rank_frequency)
    model = args.model
    output_dir = args.output
    dist = args.distribution
    arrival_pattern = args.arrival_pattern
    rps = args.rps
    time = args.time
    burst_size = args.burst_size
    burst_interval = args.burst_interval
    size = model[model.index('-') + 1:]
    names = [] # initialize with model if base requests (no adapter) are to be included
    for rank in ranks_dict:
            for idx in range(ranks_dict[rank]):
                adapter_name = f"dummy-lora-{size}-rank-{rank}-{idx}"
                names.append(adapter_name)
    normalized_weights = get_weights(len(names), dist)
    num_samples = int(rps * time)
    times = get_times(num_samples, time, arrival_pattern, burst_size, burst_interval)
    num_samples = min(num_samples, len(times))
    samples = random.choices(names, weights=normalized_weights, k=num_samples)
    df = pd.read_csv("../AzureLLMInferenceTrace_conv_1week.csv")
    df = df[(df['ContextTokens'] <= 1024) & (df['GeneratedTokens'] <= 256)]
    df = df.sample(n=num_samples)
    # df['ContextTokens'] = 512
    # df['GeneratedTokens'] = 128
    df['adapter'] = samples
    df['model'] = model
    df['timestamp'] = sorted(times)
    df['req_id'] = list(range(num_samples))
    column_order = ['req_id', 'model', 'adapter', 'ContextTokens', 'GeneratedTokens', 'timestamp']
    df = df[column_order]
    df.columns = ['req_id', 'model_dir', 'adapter_dir', 'prompt_len', 'output_len', 'req_time']
    # df["prompt_len"] = 500
    # df["output_len"] = 128
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f'{dist}_{arrival_pattern}_{rps}_{time}.csv'), index=False)
    
if __name__ == "__main__":
    main()
