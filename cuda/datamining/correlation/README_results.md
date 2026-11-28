# CUDA Correlation — Memory Model Benchmark Results

Platform: Jetson Nano. Baseline launch configuration: 256 threads per 1D block and 16 × 16 threads per 2D block. Results from the supplied nvprof logs; one reported run per configuration.

## 1. Execution time, throughput and speedup

Speedup = CPU time / GPU time. GFLOPS uses the dominant correlation-stage operation count, divided by the measured processing time.

| Memory | Dataset | GPU (s) | CPU (s) | GPU GFLOPS | CPU GFLOPS | Speedup | Validation | Max error |
|---|---|---:|---:|---:|---:|---:|---|---:|
| UVM | MINI | 0.000514 | 0.000054 | 0.062 | 0.585 | 0.11x | PASSED | 0.0 |
| UVM | SMALL | 0.011679 | 0.546944 | 10.682 | 0.228 | 46.83x | PASSED | 0.0 |
| UVM | STANDARD | 0.087526 | 4.915806 | 11.414 | 0.203 | 56.16x | PASSED | 0.0 |
| UVM | LARGE | 0.682587 | 43.185649 | 11.714 | 0.185 | 63.27x | PASSED | 0.0 |
| UVM | EXTRALARGE | 10.087010 | 459.091381 | 6.343 | 0.139 | 45.51x | PASSED | 0.0 |
| Pinned | MINI | 0.000617 | 0.000053 | 0.051 | 0.596 | 0.09x | PASSED | 0.0 |
| Pinned | SMALL | 0.011629 | 0.542153 | 10.728 | 0.230 | 46.62x | PASSED | 0.0 |
| Pinned | STANDARD | 0.087024 | 4.913155 | 11.480 | 0.203 | 56.46x | PASSED | 0.0 |
| Pinned | LARGE | 0.665383 | 43.611753 | 12.017 | 0.183 | 65.54x | PASSED | 0.0 |
| Pinned | EXTRALARGE | 8.164631 | 459.073794 | 7.837 | 0.139 | 56.23x | PASSED | 0.0 |
| Pageable | MINI | 0.000551 | 0.000054 | 0.058 | 0.592 | 0.10x | PASSED | 0.0 |
| Pageable | SMALL | 0.013371 | 0.552010 | 9.330 | 0.226 | 41.28x | PASSED | 0.0 |
| Pageable | STANDARD | 0.092503 | 4.916977 | 10.800 | 0.203 | 53.15x | PASSED | 0.0 |
| Pageable | LARGE | 0.680894 | 42.820013 | 11.743 | 0.187 | 62.89x | PASSED | 0.0 |
| Pageable | EXTRALARGE | 8.206678 | 460.042495 | 7.797 | 0.139 | 56.06x | PASSED | 0.0 |

## 2. GPU execution time comparison

| Dataset (matrix size) | UVM (s) | Pinned (s) | Pageable (s) | Fastest observed |
|---|---:|---:|---:|---|
| MINI (32 × 32) | 0.000514 | 0.000617 | 0.000551 | UVM |
| SMALL (500 × 500) | 0.011679 | 0.011629 | 0.013371 | Pinned |
| STANDARD (1000 × 1000) | 0.087526 | 0.087024 | 0.092503 | Pinned |
| LARGE (2000 × 2000) | 0.682587 | 0.665383 | 0.680894 | Pinned |
| EXTRALARGE (4000 × 4000) | 10.087010 | 8.164631 | 8.206678 | Pinned |

## 3. Individual GPU kernel execution times

All values in milliseconds (ms), from nvprof GPU activities.

| Memory | Dataset | Mean | Stddev | Normalize | Correlation | Total kernels | Correlation share of GPU activities |
|---|---|---:|---:|---:|---:|---:|---:|
| UVM | MINI | 0.006671 | 0.006878 | 0.004118 | 0.007971 | 0.025638 | 31.09% |
| UVM | SMALL | 0.065165 | 0.055637 | 0.293430 | 10.326000 | 10.740232 | 96.14% |
| UVM | STANDARD | 0.226160 | 0.194380 | 1.154300 | 84.483000 | 86.057840 | 98.17% |
| UVM | LARGE | 0.943770 | 0.840280 | 4.439800 | 673.290000 | 679.513850 | 99.08% |
| UVM | EXTRALARGE | 3.859900 | 3.853400 | 17.744000 | 10059.900000 | 10085.357300 | 99.75% |
| Pinned | MINI | 0.005834 | 0.005990 | 0.003957 | 0.007139 | 0.022920 | 26.72% |
| Pinned | SMALL | 0.064021 | 0.054956 | 0.296290 | 10.274000 | 10.689267 | 94.39% |
| Pinned | STANDARD | 0.205680 | 0.195790 | 1.146100 | 83.899000 | 85.446570 | 97.31% |
| Pinned | LARGE | 0.843560 | 0.837780 | 4.427700 | 654.890000 | 660.999040 | 98.62% |
| Pinned | EXTRALARGE | 3.267600 | 3.277700 | 17.719000 | 8127.050000 | 8151.314300 | 99.55% |
| Pageable | MINI | 0.005886 | 0.006566 | 0.004014 | 0.006926 | 0.023392 | 27.02% |
| Pageable | SMALL | 0.063474 | 0.055664 | 0.297330 | 10.277000 | 10.693468 | 94.37% |
| Pageable | STANDARD | 0.202400 | 0.194230 | 1.150600 | 83.913000 | 85.460230 | 93.50% |
| Pageable | LARGE | 0.848980 | 0.837570 | 4.427000 | 654.160000 | 660.273550 | 96.42% |
| Pageable | EXTRALARGE | 3.260900 | 3.228500 | 17.640000 | 8112.180000 | 8136.309400 | 98.89% |

## 4. Explicit memory transfer times

GPU activity durations in ms. UVM has no explicit cudaMemcpy in this implementation; its memory-management cost is not zero and is not separately measured here.

| Dataset | Pinned H2D | Pinned D2H | Pinned total | Pageable H2D | Pageable D2H | Pageable total | Pageable / Pinned |
|---|---:|---:|---:|---:|---:|---:|---:|
| MINI | 0.002969 | 0.000832 | 0.003801 | 0.001093 | 0.001144 | 0.002237 | 0.59x |
| SMALL | 0.097830 | 0.096994 | 0.194824 | 0.098675 | 0.097480 | 0.196155 | 1.01x |
| STANDARD | 0.389080 | 0.386790 | 0.775870 | 1.250000 | 3.034400 | 4.284400 | 5.52x |
| LARGE | 1.551700 | 1.516600 | 3.068300 | 6.426900 | 11.763000 | 18.189900 | 5.93x |
| EXTRALARGE | 6.226000 | 6.068900 | 12.294900 | 26.814000 | 40.451000 | 67.265000 | 5.47x |

## 5. Conclusions

- All 15 configurations passed validation with zero reported maximum absolute error.
- Pinned memory has the lowest observed GPU time in four of five datasets; UVM wins MINI.
- Peak observed throughput: 12.017 GFLOPS (Pinned, LARGE).
- Highest observed CPU speedup: 65.54x (Pinned, LARGE).
- Pinned transfers are 5.47x faster than pageable for EXTRALARGE and 5.93x for LARGE, based on summed GPU copy durations.
- The correlation kernel dominates GPU activity for large datasets (up to 99.75% for UVM). Prioritize optimizing this kernel.

## 6. Experimental limitations

- Single reported profiling run per configuration; repeat and report median and variability.
- Allocation and deallocation are outside the application GPU timer.
- Pinned and pageable timings include explicit transfers; UVM timing excludes final CPU access to the output. The comparison is not strictly end-to-end equivalent.
- GPU activity percentages include explicit copies for pinned/pageable, unlike UVM.
- API synchronization time overlaps kernel execution; do not add them together.
- The cause of the EXTRALARGE throughput drop is not established.
- No results for the proposed 128-thread / 8 × 8 configuration were supplied in these logs.

## 7. Original profiling logs

<details>
<summary>Expand complete nvprof output (all 15 runs)</summary>

```text
-------------------------------------------------UVM--------------------------------
MINI_DATASET :

==14357== NVPROF is profiling process 14357, command: ./correlation_uvm.exe
correlation (GPU):    0.000514 sec     0.062 GFLOPS
correlation (HOST):   0.000054 sec     0.585 GFLOPS
Maximum error: 0.0000000000e+00
Validation: PASSED
==14357== Profiling application: ./correlation_uvm.exe
==14357== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   31.09%  7.9710us         1  7.9710us  7.9710us  7.9710us  kernel_corr(int, int, float[32]*, float[32]*)
                   26.83%  6.8780us         1  6.8780us  6.8780us  6.8780us  kernel_stddev(int, int, float, float[32]*, float*, float*)
                   26.02%  6.6710us         1  6.6710us  6.6710us  6.6710us  kernel_mean(int, int, float, float[32]*, float*)
                   16.06%  4.1180us         1  4.1180us  4.1180us  4.1180us  kernel_normalize(int, int, float, float[32]*, float*, float*)
      API calls:   99.70%  340.25ms         4  85.062ms  79.897us  339.99ms  cudaMallocManaged
                    0.12%  411.99us         4  103.00us  87.189us  136.83us  cudaFree
                    0.09%  292.87us         4  73.217us  56.251us  115.21us  cudaLaunchKernel
                    0.06%  188.13us         4  47.032us  41.147us  61.407us  cudaDeviceSynchronize
                    0.03%  115.99us        97  1.1950us     729ns  22.032us  cuDeviceGetAttribute
                    0.00%  9.5310us         1  9.5310us  9.5310us  9.5310us  cuDeviceTotalMem
                    0.00%  4.1660us         4  1.0410us     885ns  1.3020us  cudaGetLastError
                    0.00%  4.0110us         2  2.0050us  1.5630us  2.4480us  cuDeviceGetCount
                    0.00%  2.5010us         2  1.2500us  1.0940us  1.4070us  cuDeviceGet
                    0.00%  1.9280us         1  1.9280us  1.9280us  1.9280us  cuDeviceGetName
                    0.00%  1.0930us         1  1.0930us  1.0930us  1.0930us  cuDeviceGetUui
                    
SMALL_DATASET:

==14419== NVPROF is profiling process 14419, command: ./correlation_uvm.exe
correlation (GPU):    0.011679 sec    10.682 GFLOPS
correlation (HOST):   0.546944 sec     0.228 GFLOPS
Maximum error: 0.0000000000e+00
Validation: PASSED
==14419== Profiling application: ./correlation_uvm.exe
==14419== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   96.14%  10.326ms         1  10.326ms  10.326ms  10.326ms  kernel_corr(int, int, float[500]*, float[500]*)
                    2.73%  293.43us         1  293.43us  293.43us  293.43us  kernel_normalize(int, int, float, float[500]*, float*, float*)
                    0.61%  65.165us         1  65.165us  65.165us  65.165us  kernel_mean(int, int, float, float[500]*, float*)
                    0.52%  55.637us         1  55.637us  55.637us  55.637us  kernel_stddev(int, int, float, float[500]*, float*, float*)
      API calls:   96.22%  322.87ms         4  80.717ms  90.210us  321.82ms  cudaMallocManaged
                    3.33%  11.160ms         4  2.7900ms  105.94us  10.384ms  cudaDeviceSynchronize
                    0.26%  880.64us         4  220.16us  117.35us  357.09us  cudaFree
                    0.15%  490.32us         4  122.58us  60.783us  291.15us  cudaLaunchKernel
                    0.03%  117.09us        97  1.2070us     729ns  23.334us  cuDeviceGetAttribute
                    0.00%  9.2710us         1  9.2710us  9.2710us  9.2710us  cuDeviceTotalMem
                    0.00%  4.7400us         4  1.1850us  1.1460us  1.2500us  cudaGetLastError
                    0.00%  4.5830us         2  2.2910us  1.6660us  2.9170us  cuDeviceGetCount
                    0.00%  2.7610us         2  1.3800us  1.3550us  1.4060us  cuDeviceGet
                    0.00%  1.9790us         1  1.9790us  1.9790us  1.9790us  cuDeviceGetName
                    0.00%  1.1460us         1  1.1460us  1.1460us  1.1460us  cuDeviceGetUuid

STANDARD_DATASET:
correlation (GPU):    0.087526 sec    11.414 GFLOPS
correlation (HOST):   4.915806 sec     0.203 GFLOPS
Maximum error: 0.0000000000e+00
Validation: PASSED
==14615== Profiling application: ./correlation_uvm.exe
==14615== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   98.17%  84.483ms         1  84.483ms  84.483ms  84.483ms  kernel_corr(int, int, float[1000]*, float[1000]*)
                    1.34%  1.1543ms         1  1.1543ms  1.1543ms  1.1543ms  kernel_normalize(int, int, float, float[1000]*, float*, float*)
                    0.26%  226.16us         1  226.16us  226.16us  226.16us  kernel_mean(int, int, float, float[1000]*, float*)
                    0.23%  194.38us         1  194.38us  194.38us  194.38us  kernel_stddev(int, int, float, float[1000]*, float*, float*)
      API calls:   78.79%  330.92ms         4  82.730ms  94.324us  328.08ms  cudaMallocManaged
                   20.59%  86.489ms         4  21.622ms  245.16us  84.553ms  cudaDeviceSynchronize
                    0.34%  1.4462ms         4  361.55us  116.04us  630.32us  cudaFree
                    0.24%  990.85us         4  247.71us  62.917us  786.05us  cudaLaunchKernel
                    0.03%  126.46us        97  1.3030us     677ns  28.490us  cuDeviceGetAttribute
                    0.00%  11.980us         1  11.980us  11.980us  11.980us  cuDeviceTotalMem
                    0.00%  4.4280us         4  1.1070us     886ns  1.1980us  cudaGetLastError
                    0.00%  4.1150us         2  2.0570us  1.4580us  2.6570us  cuDeviceGetCount
                    0.00%  2.6040us         2  1.3020us  1.2500us  1.3540us  cuDeviceGet
                    0.00%  2.5520us         1  2.5520us  2.5520us  2.5520us  cuDeviceGetName
                    0.00%     989ns         1     989ns     989ns     989ns  cuDeviceGetUuid
LARGE_DATASET :
correlation (GPU):    0.682587 sec    11.714 GFLOPS
correlation (HOST):  43.185649 sec     0.185 GFLOPS
Maximum error: 0.0000000000e+00
Validation: PASSED
==14666== Profiling application: ./correlation_uvm.exe
==14666== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.08%  673.29ms         1  673.29ms  673.29ms  673.29ms  kernel_corr(int, int, float[2000]*, float[2000]*)
                    0.65%  4.4398ms         1  4.4398ms  4.4398ms  4.4398ms  kernel_normalize(int, int, float, float[2000]*, float*, float*)
                    0.14%  943.77us         1  943.77us  943.77us  943.77us  kernel_mean(int, int, float, float[2000]*, float*)
                    0.12%  840.28us         1  840.28us  840.28us  840.28us  kernel_stddev(int, int, float, float[2000]*, float*, float*)
      API calls:   65.88%  679.97ms         4  169.99ms  894.75us  673.37ms  cudaDeviceSynchronize
                   33.47%  345.44ms         4  86.360ms  152.81us  334.77ms  cudaMallocManaged
                    0.39%  4.0361ms         4  1.0090ms  126.20us  1.8904ms  cudaFree
                    0.25%  2.5767ms         4  644.17us  61.043us  2.3572ms  cudaLaunchKernel
                    0.01%  114.53us        97  1.1800us     677ns  23.646us  cuDeviceGetAttribute
                    0.00%  8.4380us         1  8.4380us  8.4380us  8.4380us  cuDeviceTotalMem
                    0.00%  4.6880us         4  1.1720us  1.0940us  1.1980us  cudaGetLastError
                    0.00%  4.1150us         2  2.0570us  1.4580us  2.6570us  cuDeviceGetCount
                    0.00%  2.5520us         2  1.2760us  1.0940us  1.4580us  cuDeviceGet
                    0.00%  1.5620us         1  1.5620us  1.5620us  1.5620us  cuDeviceGetName
                    0.00%     938ns         1     938ns     938ns     938ns  cuDeviceGetUuid
EXTRALARGE_DATASET:

correlation (GPU):   10.087010 sec     6.343 GFLOPS
correlation (HOST): 459.091381 sec     0.139 GFLOPS
Maximum error: 0.0000000000e+00
Validation: PASSED
==14757== Profiling application: ./correlation_uvm.exe
==14757== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.75%  10.0599s         1  10.0599s  10.0599s  10.0599s  kernel_corr(int, int, float[4000]*, float[4000]*)
                    0.18%  17.744ms         1  17.744ms  17.744ms  17.744ms  kernel_normalize(int, int, float, float[4000]*, float*, float*)
                    0.04%  3.8599ms         1  3.8599ms  3.8599ms  3.8599ms  kernel_mean(int, int, float, float[4000]*, float*)
                    0.04%  3.8534ms         1  3.8534ms  3.8534ms  3.8534ms  kernel_stddev(int, int, float, float[4000]*, float*, float*)
      API calls:   95.90%  10.0859s         4  2.52147s  3.9099ms  10.0600s  cudaDeviceSynchronize
                    3.95%  415.16ms         4  103.79ms  163.39us  374.92ms  cudaMallocManaged
                    0.14%  14.580ms         4  3.6449ms  122.66us  7.1844ms  cudaFree
                    0.01%  1.0811ms         4  270.26us  64.428us  858.76us  cudaLaunchKernel
                    0.00%  147.82us        97  1.5230us     729ns  43.074us  cuDeviceGetAttribute
                    0.00%  17.083us         1  17.083us  17.083us  17.083us  cuDeviceTotalMem
                    0.00%  6.5120us         2  3.2560us  3.2300us  3.2820us  cuDeviceGetCount
                    0.00%  4.7920us         4  1.1980us  1.0940us  1.2500us  cudaGetLastError
                    0.00%  3.5940us         2  1.7970us  1.7190us  1.8750us  cuDeviceGet
                    0.00%  2.3960us         1  2.3960us  2.3960us  2.3960us  cuDeviceGetName
                    0.00%  1.0410us         1  1.0410us  1.0410us  1.0410us  cuDeviceGetUuid


------------------------------------ RESULT FOR THE PINNED MEMORY ---------------------------------------

MINI_DATASET :

correlation (GPU):    0.000617 sec     0.051 GFLOPS
correlation (HOST):   0.000053 sec     0.596 GFLOPS
Maximum error: 0.0000000000e+00
Validation: PASSED
==14898== Profiling application: ./correlation_pinned.exe
==14898== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   26.72%  7.1390us         1  7.1390us  7.1390us  7.1390us  kernel_corr(int, int, float[32]*, float[32]*)
                   22.42%  5.9900us         1  5.9900us  5.9900us  5.9900us  kernel_stddev(int, int, float, float[32]*, float*, float*)
                   21.83%  5.8340us         1  5.8340us  5.8340us  5.8340us  kernel_mean(int, int, float, float[32]*, float*)
                   14.81%  3.9570us         1  3.9570us  3.9570us  3.9570us  kernel_normalize(int, int, float, float[32]*, float*, float*)
                   11.11%  2.9690us         1  2.9690us  2.9690us  2.9690us  [CUDA memcpy HtoD]
                    3.11%     832ns         1     832ns     832ns     832ns  [CUDA memcpy DtoH]
      API calls:   99.37%  318.33ms         4  79.583ms  13.906us  318.29ms  cudaMalloc
                    0.27%  855.48us         2  427.74us  16.094us  839.39us  cudaHostAlloc
                    0.10%  331.26us         2  165.63us  40.157us  291.10us  cudaMemcpy
                    0.07%  224.59us         2  112.29us  29.584us  195.00us  cudaFreeHost
                    0.07%  212.92us         4  53.230us  15.990us  146.62us  cudaFree
                    0.07%  212.87us         4  53.216us  39.897us  84.584us  cudaLaunchKernel
                    0.04%  114.85us        97  1.1830us     625ns  25.261us  cuDeviceGetAttribute
                    0.01%  37.656us         4  9.4140us  7.8120us  12.708us  cudaDeviceSynchronize
                    0.00%  11.094us         1  11.094us  11.094us  11.094us  cuDeviceTotalMem
                    0.00%  4.8960us         2  2.4480us  1.9790us  2.9170us  cuDeviceGetCount
                    0.00%  3.3880us         4     847ns     730ns  1.0420us  cudaGetLastError
                    0.00%  2.8650us         2  1.4320us  1.4070us  1.4580us  cuDeviceGet
                    0.00%  1.9280us         1  1.9280us  1.9280us  1.9280us  cuDeviceGetName
                    0.00%     937ns         1     937ns     937ns     937ns  cuDeviceGetUuid
SMALL_DATASET:

correlation (GPU):    0.011629 sec    10.728 GFLOPS
correlation (HOST):   0.542153 sec     0.230 GFLOPS
Maximum error: 0.0000000000e+00
Validation: PASSED
==15010== Profiling application: ./correlation_pinned.exe
==15010== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   94.39%  10.274ms         1  10.274ms  10.274ms  10.274ms  kernel_corr(int, int, float[500]*, float[500]*)
                    2.72%  296.29us         1  296.29us  296.29us  296.29us  kernel_normalize(int, int, float, float[500]*, float*, float*)
                    0.90%  97.830us         1  97.830us  97.830us  97.830us  [CUDA memcpy HtoD]
                    0.89%  96.994us         1  96.994us  96.994us  96.994us  [CUDA memcpy DtoH]
                    0.59%  64.021us         1  64.021us  64.021us  64.021us  kernel_mean(int, int, float, float[500]*, float*)
                    0.50%  54.956us         1  54.956us  54.956us  54.956us  kernel_stddev(int, int, float, float[500]*, float*, float*)
      API calls:   95.76%  322.89ms         4  80.722ms  18.281us  322.11ms  cudaMalloc
                    3.19%  10.761ms         4  2.6902ms  71.460us  10.297ms  cudaDeviceSynchronize
                    0.48%  1.6236ms         2  811.81us  794.44us  829.18us  cudaHostAlloc
                    0.17%  573.86us         2  286.93us  176.46us  397.40us  cudaMemcpy
                    0.16%  527.46us         2  263.73us  248.44us  279.02us  cudaFreeHost
                    0.12%  400.89us         4  100.22us  30.470us  172.03us  cudaFree
                    0.08%  263.60us         4  65.899us  44.636us  98.908us  cudaLaunchKernel
                    0.03%  113.33us        97  1.1680us     677ns  22.605us  cuDeviceGetAttribute
                    0.00%  8.9580us         1  8.9580us  8.9580us  8.9580us  cuDeviceTotalMem
                    0.00%  4.7920us         4  1.1980us  1.0940us  1.3540us  cudaGetLastError
                    0.00%  4.3750us         2  2.1870us  1.7190us  2.6560us  cuDeviceGetCount
                    0.00%  2.6560us         2  1.3280us  1.2500us  1.4060us  cuDeviceGet
                    0.00%  1.9270us         1  1.9270us  1.9270us  1.9270us  cuDeviceGetName
                    0.00%     989ns         1     989ns     989ns     989ns  cuDeviceGetUuid
                
                
    STANDARD_DATASET:
    
    correlation (GPU):    0.087024 sec    11.480 GFLOPS
correlation (HOST):   4.913155 sec     0.203 GFLOPS
Maximum error: 0.0000000000e+00
Validation: PASSED
==15067== Profiling application: ./correlation_pinned.exe
==15067== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   97.31%  83.899ms         1  83.899ms  83.899ms  83.899ms  kernel_corr(int, int, float[1000]*, float[1000]*)
                    1.33%  1.1461ms         1  1.1461ms  1.1461ms  1.1461ms  kernel_normalize(int, int, float, float[1000]*, float*, float*)
                    0.45%  389.08us         1  389.08us  389.08us  389.08us  [CUDA memcpy HtoD]
                    0.45%  386.79us         1  386.79us  386.79us  386.79us  [CUDA memcpy DtoH]
                    0.24%  205.68us         1  205.68us  205.68us  205.68us  kernel_mean(int, int, float, float[1000]*, float*)
                    0.23%  195.79us         1  195.79us  195.79us  195.79us  kernel_stddev(int, int, float, float[1000]*, float*, float*)
      API calls:   77.86%  331.01ms         4  82.752ms  19.115us  327.71ms  cudaMalloc
                   20.12%  85.552ms         4  21.388ms  219.17us  83.931ms  cudaDeviceSynchronize
                    1.25%  5.2974ms         2  2.6487ms  2.6409ms  2.6565ms  cudaHostAlloc
                    0.28%  1.1861ms         2  593.03us  470.37us  715.69us  cudaMemcpy
                    0.27%  1.1524ms         2  576.21us  543.97us  608.45us  cudaFreeHost
                    0.13%  550.48us         4  137.62us  35.417us  227.92us  cudaFree
                    0.06%  249.74us         4  62.435us  46.042us  101.72us  cudaLaunchKernel
                    0.03%  116.20us        97  1.1970us     677ns  23.437us  cuDeviceGetAttribute
                    0.00%  9.2190us         1  9.2190us  9.2190us  9.2190us  cuDeviceTotalMem
                    0.00%  4.4790us         4  1.1190us     989ns  1.3540us  cudaGetLastError
                    0.00%  4.4280us         2  2.2140us  1.6670us  2.7610us  cuDeviceGetCount
                    0.00%  2.8130us         2  1.4060us  1.1460us  1.6670us  cuDeviceGet
                    0.00%  1.8760us         1  1.8760us  1.8760us  1.8760us  cuDeviceGetName
                    0.00%  1.0410us         1  1.0410us  1.0410us  1.0410us  cuDeviceGetUuid
                    
                    
LARGE_DATASET:

correlation (GPU):    0.665383 sec    12.017 GFLOPS
correlation (HOST):  43.611753 sec     0.183 GFLOPS
Maximum error: 0.0000000000e+00
Validation: PASSED
==15158== Profiling application: ./correlation_pinned.exe
==15158== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   98.62%  654.89ms         1  654.89ms  654.89ms  654.89ms  kernel_corr(int, int, float[2000]*, float[2000]*)
                    0.67%  4.4277ms         1  4.4277ms  4.4277ms  4.4277ms  kernel_normalize(int, int, float, float[2000]*, float*, float*)
                    0.23%  1.5517ms         1  1.5517ms  1.5517ms  1.5517ms  [CUDA memcpy HtoD]
                    0.23%  1.5166ms         1  1.5166ms  1.5166ms  1.5166ms  [CUDA memcpy DtoH]
                    0.13%  843.56us         1  843.56us  843.56us  843.56us  kernel_mean(int, int, float, float[2000]*, float*)
                    0.13%  837.78us         1  837.78us  837.78us  837.78us  kernel_stddev(int, int, float, float[2000]*, float*, float*)
      API calls:   63.98%  661.42ms         4  165.35ms  872.72us  654.96ms  cudaDeviceSynchronize
                   33.15%  342.68ms         4  85.671ms  20.990us  332.47ms  cudaMalloc
                    1.94%  20.105ms         2  10.053ms  10.032ms  10.073ms  cudaHostAlloc
                    0.44%  4.5043ms         2  2.2522ms  2.2169ms  2.2874ms  cudaFreeHost
                    0.35%  3.6459ms         2  1.8230ms  1.6896ms  1.9563ms  cudaMemcpy
                    0.10%  993.40us         4  248.35us  49.271us  398.34us  cudaFree
                    0.03%  276.93us         4  69.233us  46.094us  119.27us  cudaLaunchKernel
                    0.01%  109.33us        97  1.1270us     625ns  22.396us  cuDeviceGetAttribute
                    0.00%  10.105us         1  10.105us  10.105us  10.105us  cuDeviceTotalMem
                    0.00%  4.6870us         4  1.1710us     989ns  1.4580us  cudaGetLastError
                    0.00%  4.4790us         2  2.2390us  1.8230us  2.6560us  cuDeviceGetCount
                    0.00%  2.5520us         2  1.2760us  1.1460us  1.4060us  cuDeviceGet
                    0.00%  2.0840us         1  2.0840us  2.0840us  2.0840us  cuDeviceGetName
                    0.00%     833ns         1     833ns     833ns     833ns  cuDeviceGetUuid

EXTRALARGE_DATASET:

correlation (GPU):    8.164631 sec     7.837 GFLOPS
correlation (HOST): 459.073794 sec     0.139 GFLOPS
Maximum error: 0.0000000000e+00
Validation: PASSED
==15251== Profiling application: ./correlation_pinned.exe
==15251== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.55%  8.12705s         1  8.12705s  8.12705s  8.12705s  kernel_corr(int, int, float[4000]*, float[4000]*)
                    0.22%  17.719ms         1  17.719ms  17.719ms  17.719ms  kernel_normalize(int, int, float, float[4000]*, float*, float*)
                    0.08%  6.2260ms         1  6.2260ms  6.2260ms  6.2260ms  [CUDA memcpy HtoD]
                    0.07%  6.0689ms         1  6.0689ms  6.0689ms  6.0689ms  [CUDA memcpy DtoH]
                    0.04%  3.2777ms         1  3.2777ms  3.2777ms  3.2777ms  kernel_stddev(int, int, float, float[4000]*, float*, float*)
                    0.04%  3.2676ms         1  3.2676ms  3.2676ms  3.2676ms  kernel_mean(int, int, float, float[4000]*, float*)
      API calls:   94.13%  8.15150s         4  2.03787s  3.2941ms  8.12715s  cudaDeviceSynchronize
                    4.62%  400.11ms         4  100.03ms  18.385us  362.27ms  cudaMalloc
                    0.92%  79.308ms         2  39.654ms  39.642ms  39.666ms  cudaHostAlloc
                    0.16%  13.993ms         2  6.9964ms  6.9177ms  7.0752ms  cudaFreeHost
                    0.15%  12.806ms         2  6.4030ms  6.2052ms  6.6008ms  cudaMemcpy
                    0.02%  2.0153ms         4  503.82us  49.949us  959.55us  cudaFree
                    0.00%  282.04us         4  70.509us  51.980us  112.50us  cudaLaunchKernel
                    0.00%  113.44us        97  1.1690us     625ns  23.646us  cuDeviceGetAttribute
                    0.00%  9.8430us         1  9.8430us  9.8430us  9.8430us  cuDeviceTotalMem
                    0.00%  5.3640us         4  1.3410us  1.1450us  1.6150us  cudaGetLastError
                    0.00%  4.3230us         2  2.1610us  1.7710us  2.5520us  cuDeviceGetCount
                    0.00%  2.7610us         2  1.3800us  1.3020us  1.4590us  cuDeviceGet
                    0.00%  1.8750us         1  1.8750us  1.8750us  1.8750us  cuDeviceGetName
                    0.00%     885ns         1     885ns     885ns     885ns  cuDeviceGetUuid
--------------------------------------------- PEAGEABLE----------------------------------

MINI_DATASET:

correlation (GPU):    0.000551 sec     0.058 GFLOPS
correlation (HOST):   0.000054 sec     0.592 GFLOPS
Maximum error: 0.0000000000e+00
Validation: PASSED
==15355== Profiling application: ./correlation_pageable.exe
==15355== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   27.02%  6.9260us         1  6.9260us  6.9260us  6.9260us  kernel_corr(int, int, float[32]*, float[32]*)
                   25.62%  6.5660us         1  6.5660us  6.5660us  6.5660us  kernel_stddev(int, int, float, float[32]*, float*, float*)
                   22.97%  5.8860us         1  5.8860us  5.8860us  5.8860us  kernel_mean(int, int, float, float[32]*, float*)
                   15.66%  4.0140us         1  4.0140us  4.0140us  4.0140us  kernel_normalize(int, int, float, float[32]*, float*, float*)
                    4.46%  1.1440us         1  1.1440us  1.1440us  1.1440us  [CUDA memcpy DtoH]
                    4.26%  1.0930us         1  1.0930us  1.0930us  1.0930us  [CUDA memcpy HtoD]
      API calls:   99.73%  321.68ms         4  80.420ms  13.594us  321.63ms  cudaMalloc
                    0.07%  212.03us         4  53.008us  39.584us  80.834us  cudaLaunchKernel
                    0.06%  202.92us         4  50.729us  14.948us  144.48us  cudaFree
                    0.05%  163.44us         4  40.859us  8.9580us  135.31us  cudaDeviceSynchronize
                    0.05%  152.09us         2  76.043us  75.470us  76.616us  cudaMemcpy
                    0.04%  122.25us        97  1.2600us     729ns  22.188us  cuDeviceGetAttribute
                    0.00%  8.7510us         1  8.7510us  8.7510us  8.7510us  cuDeviceTotalMem
                    0.00%  5.1050us         2  2.5520us  2.0320us  3.0730us  cuDeviceGetCount
                    0.00%  4.2700us         4  1.0670us     937ns  1.1460us  cudaGetLastError
                    0.00%  2.7090us         2  1.3540us  1.3540us  1.3550us  cuDeviceGet
                    0.00%  1.8750us         1  1.8750us  1.8750us  1.8750us  cuDeviceGetName
                    0.00%  1.0940us         1  1.0940us  1.0940us  1.0940us  cuDeviceGetUuid
                    
                    

SMALL DATA_SET :

correlation (GPU):    0.013371 sec     9.330 GFLOPS
correlation (HOST):   0.552010 sec     0.226 GFLOPS
Maximum error: 0.0000000000e+00
Validation: PASSED
==15432== Profiling application: ./correlation_pageable.exe
==15432== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   94.37%  10.277ms         1  10.277ms  10.277ms  10.277ms  kernel_corr(int, int, float[500]*, float[500]*)
                    2.73%  297.33us         1  297.33us  297.33us  297.33us  kernel_normalize(int, int, float, float[500]*, float*, float*)
                    0.91%  98.675us         1  98.675us  98.675us  98.675us  [CUDA memcpy HtoD]
                    0.90%  97.480us         1  97.480us  97.480us  97.480us  [CUDA memcpy DtoH]
                    0.58%  63.474us         1  63.474us  63.474us  63.474us  kernel_mean(int, int, float, float[500]*, float*)
                    0.51%  55.664us         1  55.664us  55.664us  55.664us  kernel_stddev(int, int, float, float[500]*, float*, float*)
      API calls:   95.92%  326.63ms         4  81.658ms  18.639us  325.85ms  cudaMalloc
                    3.22%  10.949ms         4  2.7373ms  72.059us  10.300ms  cudaDeviceSynchronize
                    0.62%  2.1251ms         2  1.0625ms  467.24us  1.6578ms  cudaMemcpy
                    0.12%  422.10us         4  105.52us  32.124us  173.22us  cudaFree
                    0.08%  264.70us         4  66.175us  44.204us  109.03us  cudaLaunchKernel
                    0.03%  114.39us        97  1.1790us     676ns  23.117us  cuDeviceGetAttribute
                    0.00%  9.0600us         1  9.0600us  9.0600us  9.0600us  cuDeviceTotalMem
                    0.00%  4.6860us         2  2.3430us  1.5100us  3.1760us  cuDeviceGetCount
                    0.00%  4.0080us         4  1.0020us     781ns  1.1970us  cudaGetLastError
                    0.00%  2.8110us         2  1.4050us  1.3530us  1.4580us  cuDeviceGet
                    0.00%  1.9270us         1  1.9270us  1.9270us  1.9270us  cuDeviceGetName
                    0.00%  1.0420us         1  1.0420us  1.0420us  1.0420us  cuDeviceGetUuid

STANDARD_DATASET:

correlation (GPU):    0.092503 sec    10.800 GFLOPS
correlation (HOST):   4.916977 sec     0.203 GFLOPS
Maximum error: 0.0000000000e+00
Validation: PASSED
==15508== Profiling application: ./correlation_pageable.exe
==15508== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   93.50%  83.913ms         1  83.913ms  83.913ms  83.913ms  kernel_corr(int, int, float[1000]*, float[1000]*)
                    3.38%  3.0344ms         1  3.0344ms  3.0344ms  3.0344ms  [CUDA memcpy DtoH]
                    1.39%  1.2500ms         1  1.2500ms  1.2500ms  1.2500ms  [CUDA memcpy HtoD]
                    1.28%  1.1506ms         1  1.1506ms  1.1506ms  1.1506ms  kernel_normalize(int, int, float, float[1000]*, float*, float*)
                    0.23%  202.40us         1  202.40us  202.40us  202.40us  kernel_mean(int, int, float, float[1000]*, float*)
                    0.22%  194.23us         1  194.23us  194.23us  194.23us  kernel_stddev(int, int, float, float[1000]*, float*, float*)
      API calls:   78.05%  331.13ms         4  82.783ms  19.891us  327.84ms  cudaMalloc
                   20.17%  85.581ms         4  21.395ms  212.40us  83.945ms  cudaDeviceSynchronize
                    1.55%  6.5895ms         2  3.2948ms  1.8734ms  4.7162ms  cudaMemcpy
                    0.13%  538.99us         4  134.75us  34.315us  219.64us  cudaFree
                    0.07%  286.60us         4  71.649us  44.260us  133.56us  cudaLaunchKernel
                    0.03%  118.98us        97  1.2260us     729ns  24.265us  cuDeviceGetAttribute
                    0.00%  9.8420us         1  9.8420us  9.8420us  9.8420us  cuDeviceTotalMem
                    0.00%  4.7910us         4  1.1970us     989ns  1.4580us  cudaGetLastError
                    0.00%  4.5300us         2  2.2650us  1.7180us  2.8120us  cuDeviceGetCount
                    0.00%  2.5520us         2  1.2760us  1.0940us  1.4580us  cuDeviceGet
                    0.00%  1.7710us         1  1.7710us  1.7710us  1.7710us  cuDeviceGetName
                    0.00%     885ns         1     885ns     885ns     885ns  cuDeviceGetUuid

LARGE_DATASET:

correlation (GPU):    0.680894 sec    11.743 GFLOPS
correlation (HOST):  42.820013 sec     0.187 GFLOPS
Maximum error: 0.0000000000e+00
Validation: PASSED
==15559== Profiling application: ./correlation_pageable.exe
==15559== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   96.42%  654.16ms         1  654.16ms  654.16ms  654.16ms  kernel_corr(int, int, float[2000]*, float[2000]*)
                    1.73%  11.763ms         1  11.763ms  11.763ms  11.763ms  [CUDA memcpy DtoH]
                    0.95%  6.4269ms         1  6.4269ms  6.4269ms  6.4269ms  [CUDA memcpy HtoD]
                    0.65%  4.4270ms         1  4.4270ms  4.4270ms  4.4270ms  kernel_normalize(int, int, float, float[2000]*, float*, float*)
                    0.13%  848.98us         1  848.98us  848.98us  848.98us  kernel_mean(int, int, float, float[2000]*, float*)
                    0.12%  837.57us         1  837.57us  837.57us  837.57us  kernel_stddev(int, int, float, float[2000]*, float*, float*)
      API calls:   63.34%  660.41ms         4  165.10ms  858.55us  654.20ms  cudaDeviceSynchronize
                   34.61%  360.84ms         4  90.209ms  19.893us  350.64ms  cudaMalloc
                    1.93%  20.162ms         2  10.081ms  7.1146ms  13.048ms  cudaMemcpy
                    0.08%  794.20us         4  198.55us  39.264us  316.93us  cudaFree
                    0.03%  278.13us         4  69.532us  47.127us  118.99us  cudaLaunchKernel
                    0.01%  118.42us        97  1.2200us     677ns  23.173us  cuDeviceGetAttribute
                    0.00%  8.5400us         1  8.5400us  8.5400us  8.5400us  cuDeviceTotalMem
                    0.00%  4.4800us         4  1.1200us     886ns  1.3020us  cudaGetLastError
                    0.00%  4.4260us         2  2.2130us  1.8740us  2.5520us  cuDeviceGetCount
                    0.00%  2.7080us         2  1.3540us  1.1980us  1.5100us  cuDeviceGet
                    0.00%  2.3950us         1  2.3950us  2.3950us  2.3950us  cuDeviceGetName
                    0.00%     989ns         1     989ns     989ns     989ns  cuDeviceGetUuid
                    
EXTRALARGE_DATASET :

correlation (GPU):    8.206678 sec     7.797 GFLOPS
correlation (HOST): 460.042495 sec     0.139 GFLOPS
Maximum error: 0.0000000000e+00
Validation: PASSED
==15651== Profiling application: ./correlation_pageable.exe
==15651== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   98.89%  8.11218s         1  8.11218s  8.11218s  8.11218s  kernel_corr(int, int, float[4000]*, float[4000]*)
                    0.49%  40.451ms         1  40.451ms  40.451ms  40.451ms  [CUDA memcpy DtoH]
                    0.33%  26.814ms         1  26.814ms  26.814ms  26.814ms  [CUDA memcpy HtoD]
                    0.22%  17.640ms         1  17.640ms  17.640ms  17.640ms  kernel_normalize(int, int, float, float[4000]*, float*, float*)
                    0.04%  3.2609ms         1  3.2609ms  3.2609ms  3.2609ms  kernel_mean(int, int, float, float[4000]*, float*)
                    0.04%  3.2285ms         1  3.2285ms  3.2285ms  3.2285ms  kernel_stddev(int, int, float, float[4000]*, float*, float*)
      API calls:   94.36%  8.13781s         4  2.03445s  3.2542ms  8.11361s  cudaDeviceSynchronize
                    4.81%  415.17ms         4  103.79ms  18.287us  377.27ms  cudaMalloc
                    0.79%  68.489ms         2  34.244ms  27.505ms  40.984ms  cudaMemcpy
                    0.02%  2.0478ms         4  511.95us  49.741us  947.21us  cudaFree
                    0.00%  333.48us         4  83.370us  51.942us  165.41us  cudaLaunchKernel
                    0.00%  119.98us        97  1.2360us     729ns  24.278us  cuDeviceGetAttribute
                    0.00%  8.6480us         1  8.6480us  8.6480us  8.6480us  cuDeviceTotalMem
                    0.00%  4.8450us         4  1.2110us  1.0420us  1.5110us  cudaGetLastError
                    0.00%  4.3760us         2  2.1880us  1.5110us  2.8650us  cuDeviceGetCount
                    0.00%  2.6570us         2  1.3280us  1.1460us  1.5110us  cuDeviceGet
                    0.00%  2.1360us         1  2.1360us  2.1360us  2.1360us  cuDeviceGetName
                    0.00%  1.0420us         1  1.0420us  1.0420us  1.0420us  cuDeviceGetUuid
```

</details>
