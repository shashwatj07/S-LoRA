The notebooks in this folder are for testing out the system allocation algorithm. 

## system_algo.ipynb
Contains the previous iterations of the algorithm for record keeping, including the score based methods.

## test_allocations.ipynb
- You must have the requirements from `requirements.txt` installed for testing out the allocation.
- Takes a trace file as input (ex: `skew_uniform_3.0_300.csv`), runs the algo and outputs the allocations at every step to `allocation_log.txt`. Remember to clear the previous allocations. Adjust step size, trace, number of servers and the algo from this notebook.
- More trace files can be found at [LoRA Serve Traces](https://microsoftapc-my.sharepoint.com/:f:/g/personal/t-sarun_microsoft_com/EgZn6yrAeMZGtlYtZgA-31oB8xEcpBm3QFwougThkxGmWw?e=Tx2lpv) or [alt link](https://microsoftapc-my.sharepoint.com/:f:/r/personal/t-sarun_microsoft_com/Documents/LoRA%20Serve?csf=1&web=1&e=LChi4q)
- The traces were generated using `../generate_trace.py` 