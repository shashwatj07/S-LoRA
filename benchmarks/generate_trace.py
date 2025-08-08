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
        return random.shuffle(dist)
    elif dist == "skew":
        return [1.0 for _ in range(20)] + [4.0 for _ in range(5)]
    elif dist == "uniform":
        return [1.0 / size for _ in range(size)]

def get_times(size, end_time, dist):
    if dist == "uniform":
        return np.random.uniform(0, end_time, size)
    elif dist == "even":
        return np.linspace(0, end_time, size)

def main():
    parser = argparse.ArgumentParser(description="Generate Traces")
    parser.add_argument("--output", "-o", type=str, default="./traces",
                        help="Path to write traces")
    parser.add_argument("--model", "-m", type=str, required=True, help="HF model mdentifier")
    parser.add_argument("--rank-frequency", "-r", type=parse_key_value, nargs="+", required=True,
                        help="Pass rank=frequency pairs like 8=3 16=4 128=6")
    parser.add_argument("--distribution", "-d", type=str, default="pareto", help="Popularity distribution of adapters")
    parser.add_argument("--arrival-pattern", "-a", type=str, default="uniform", help="Arrival pattern of requests")
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
    size = model[model.index('-') + 1:]
    names = [] # initialize with model if base requests (no adapter) are to be included
    for rank in ranks_dict:
            for idx in range(ranks_dict[rank]):
                adapter_name = f"dummy-lora-{size}-rank-{rank}-{idx}"
                names.append(adapter_name)
    normalized_weights = get_weights(len(names), dist)
    num_samples = int(rps * time)
    samples = random.choices(names, weights=normalized_weights, k=num_samples)
    times = get_times(num_samples, time, arrival_pattern)
    df = pd.read_csv("/home/t-shajaiswal/AzurePublicDataset/AzureLLMInferenceTrace_conv_1week.csv")
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
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f'{dist}_{arrival_pattern}_{rps}_{time}.csv'), index=False)
    
if __name__ == "__main__":
    main()
