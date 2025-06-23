# Requirements
* CUDA 11.8

# Install
```
conda create -n slora python=3.9
conda activate slora
pip install -r requirements.txt
pip install -e .
```
Check that we have triton==2.1.0

# Example Run
```
cd benchmarks
python launch_server.py --num-adapter 1 --device debug --backend dm
python run_exp.py --debug --backend dm --suite swap
```

# Example Run With TP
```
cd benchmarks
python launch_server.py --num-adapter 1 --num-token 10000 --model-setting S1 --device h100 --backend dm --tp 8
python run_exp.py --mode synthetic --model-setting S1 --output output.jsonl --suite default --backend dm
```

# Test Correctness
```
cd test/test_e2e
python launch_server.py
python run_exp.py
```

# Plots
```
cd benchmarks/plot
python plot_main_synthetic.py
python plot_main_real.py
python plot_ablation_abort.py
python plot_cluster_ablation.py
```
