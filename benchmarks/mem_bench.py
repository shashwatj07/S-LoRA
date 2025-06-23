import torch
import time
import os
import numpy as np

def benchmark_transfer(src_device, dst_device, tensor_size_gb, iterations):
    # Allocate tensor on source device
    size = (tensor_size_gb * 1024 * 1024 * 1024) // 4  # float32 -> 4 bytes
    size = int(size)
    file_name = f"tensor_{tensor_size_gb}GB.pt"
    if src_device == "disk":
        if not os.path.exists(file_name):
            tensor = torch.randn(size, dtype=torch.float32)
            torch.save(tensor, file_name)
            # tensor.cpu().numpy().tofile(file_name)
    
    src_tensor = torch.randn(size, dtype=torch.float32, device=src_device if src_device != "disk" else "cpu")
    
    # Warm-up
    dst_tensor = torch.randn(size, dtype=torch.float32, device=dst_device)
    dst_tensor.sum().item()

    total_time = 0
    for _ in range(iterations):
        dst_tensor.normal_()
        start_time = time.time()
        dst_tensor.sum().item()
        end_time = time.time()
        total_time += end_time - start_time
    sum_time  = total_time / iterations

    # torch.cuda.synchronize()

    total_time = 0
    # Measure latency and bandwidth
    for i in range(iterations + 1):
        src_tensor.normal_()
        start_time = time.time()
        if src_device == "disk":
            src_tensor = torch.load(file_name, mmap=True)
            # src_tensor = torch.from_numpy(np.memmap(file_name, dtype="float32", mode="r").copy())
            # dst_tensor.copy_(src_tensor, non_blocking=False)
            dst_tensor.copy_(src_tensor, non_blocking=False)
        else:
            dst_tensor.copy_(src_tensor, non_blocking=False)
        dst_tensor.sum().item()
        end_time = time.time()
        if i > 0:
            total_time += end_time - start_time - sum_time
        
        # tensor = tensor.to(src_device)
        # torch.cuda.synchronize()

    avg_time = total_time / iterations
    bandwidth = tensor_size_gb / avg_time  # GB/s
    with open("output.csv", "a") as f:
        f.write(f"{src_device},{dst_device},{tensor_size_gb},{avg_time * 1e3},{bandwidth}\n")
    # print(f"Tensor Size: {tensor_size_gb} GB", end=", ")
    # print(f"Transfer Latency: {avg_time * 1e3:.3f} ms", end=", ")
    # print(f"Effective Bandwidth: {bandwidth:.2f} GB/s")

if __name__ == "__main__":
    src = "disk"
    dst = "cuda:3"
    sizes_gb = [2**i for i in range(-4, 6, 1)]
    for gb in sizes_gb:
        benchmark_transfer(src, dst, tensor_size_gb=gb, iterations=5)
