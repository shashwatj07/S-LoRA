import torch, time

size = 1024 * 1024 * 1024  # 1GB
src = torch.empty(size, dtype=torch.float32).pin_memory()
dst = torch.empty(size, dtype=torch.float32, device='cuda:0')

torch.cuda.synchronize()
start = time.perf_counter()
dst.copy_(src, non_blocking=True)
torch.cuda.synchronize()
end = time.perf_counter()

bandwidth = (size * 4 / 1e9) / (end - start)  # float32 = 4 bytes
print(f"CPU→GPU Bandwidth: {bandwidth:.2f} GB/s")
