
#include <stdlib.h>
#include <chrono>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

extern "C" {
    #include <polybench.h>
}

#include "correlation.h"
static 
void gpuErrchk(cudaError_t err)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static
void init_array (int m,
		 int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,M,N,m,n))
{
  int i, j;

  *float_n = 1.2;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      data[i][j] = ((DATA_TYPE) i*j) / M;
}

static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m))

{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      fprintf (stderr, DATA_PRINTF_MODIFIER, symmat[i][j]);
      if ((i * m + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}
/* CUDA kernels implement the four stages of the correlation algorithm. */
__global__ void kernel_mean(
    int m,
    int n,
    DATA_TYPE float_n,
    DATA_TYPE (*data)[N],
    DATA_TYPE *mean)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < m)
    {
        DATA_TYPE sum = 0.0;

        for (int i = 0; i < n; i++)
        {
            sum += data[i][j];
        }

        mean[j] = sum / float_n;
    }
}

__global__ void kernel_stddev(
    int m,
    int n,
    DATA_TYPE float_n,
    DATA_TYPE (*data)[N],
    DATA_TYPE *mean,
    DATA_TYPE *stddev)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    DATA_TYPE eps = 0.1f;

    if (j < m)
    {
        DATA_TYPE sum = 0.0;

        for (int i = 0; i < n; i++)
        {
            DATA_TYPE diff = data[i][j] - mean[j];
            sum += diff * diff;
        }

        stddev[j] = sqrt(sum / float_n);

        if (stddev[j] <= eps)
            stddev[j] = 1.0;
    }
}
__global__ void kernel_normalize(
    int m,
    int n,
    DATA_TYPE float_n,
    DATA_TYPE (*data)[N],
    DATA_TYPE *mean,
    DATA_TYPE *stddev)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < m)
    {
        data[i][j] -= mean[j];

        data[i][j] /= sqrt(float_n) * stddev[j];
    }
}

__global__ void kernel_corr(
    int m,
    int n,
    DATA_TYPE (*data)[N],
    DATA_TYPE (*symmat)[M])
{
    int j2 = blockIdx.x * blockDim.x + threadIdx.x;
    int j1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (j1 < m && j2 < m)
    {
        if (j1 == j2)
        {
            symmat[j1][j2] = 1.0;
        }
        else if (j1 < j2)
        {
            DATA_TYPE sum = 0.0;

            for (int i = 0; i < n; i++)
            {
                sum += data[i][j1] * data[i][j2];
            }

            symmat[j1][j2] = sum;
            symmat[j2][j1] = sum;
        }
    }
}
/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_correlation_gpu(int m, int n,
			DATA_TYPE float_n,
			DATA_TYPE POLYBENCH_2D(data,M,N,m,n),
			DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m),
			DATA_TYPE POLYBENCH_1D(mean,M,m),
			DATA_TYPE POLYBENCH_1D(stddev,M,m))
{
#define sqrt_of_array_cell(x,j) sqrt(x[j])

  /* Determine mean of column vectors of input data matrix */
int threadsPerBlock = 256;
int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;
kernel_mean<<<blocksPerGrid, threadsPerBlock>>>(m, n, float_n, data, mean);
gpuErrchk(cudaGetLastError());
gpuErrchk(cudaDeviceSynchronize());

    /* Determine standard deviations of column vectors of data matrix. */
kernel_stddev<<<blocksPerGrid, threadsPerBlock>>>(m, n, float_n, data, mean, stddev);
gpuErrchk(cudaGetLastError());
gpuErrchk(cudaDeviceSynchronize());
    

    /* Center and reduce the column vectors. */
dim3 block(16, 16);
dim3 grid((m + block.x - 1) / block.x,(n + block.y - 1) / block.y);
kernel_normalize<<<grid, block>>>(m, n, float_n, data, mean, stddev);
gpuErrchk(cudaGetLastError());
gpuErrchk(cudaDeviceSynchronize());

    
    /* Calculate the m * m correlation matrix. */
dim3 blockCorr(16, 16);
dim3 gridCorr((m + blockCorr.x - 1) / blockCorr.x,(m + blockCorr.y - 1) / blockCorr.y);
kernel_corr<<<gridCorr, blockCorr>>>(m, n, data, symmat);
gpuErrchk(cudaGetLastError());
gpuErrchk(cudaDeviceSynchronize());
}

static
void kernel_correlation_host(int m, int n,
			DATA_TYPE float_n,
			DATA_TYPE POLYBENCH_2D(data,M,N,m,n),
			DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m),
			DATA_TYPE POLYBENCH_1D(mean,M,m),
			DATA_TYPE POLYBENCH_1D(stddev,M,m))
{
  int i, j, j1, j2;

  DATA_TYPE eps = 0.1f;

#define sqrt_of_array_cell(x,j) sqrt(x[j])

  /* Determine mean of column vectors of input data matrix */
    for (j = 0; j < _PB_M; j++)
      {
        mean[j] = 0.0;
	for (i = 0; i < _PB_N; i++)
	  mean[j] += data[i][j];
	mean[j] /= float_n;
      }
    /* Determine standard deviations of column vectors of data matrix. */
    for (j = 0; j < _PB_M; j++)
      {
        stddev[j] = 0.0;
	for (i = 0; i < _PB_N; i++)
	  stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
	stddev[j] /= float_n;
	stddev[j] = sqrt_of_array_cell(stddev, j);
	/* The following in an inelegant but usual way to handle
	   near-zero std. dev. values, which below would cause a zero-
	   divide. */
	stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
      }
    
    /* Center and reduce the column vectors. */
    for (i = 0; i < _PB_N; i++)
      for (j = 0; j < _PB_M; j++)
	{
          data[i][j] -= mean[j];
          data[i][j] /= sqrt(float_n) * stddev[j];
	}
    
    /* Calculate the m * m correlation matrix. */
    for (j1 = 0; j1 < _PB_M-1; j1++)
      {
        symmat[j1][j1] = 1.0;
	for (j2 = j1+1; j2 < _PB_M; j2++)
	  {
            symmat[j1][j2] = 0.0;
	    for (i = 0; i < _PB_N; i++)
	      symmat[j1][j2] += (data[i][j1] * data[i][j2]);
	    symmat[j2][j1] = symmat[j1][j2];
          }
      }
  symmat[_PB_M-1][_PB_M-1] = 1.0;
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE float_n;
/* Device memory */
DATA_TYPE (*data_d)[N];
DATA_TYPE (*symmat_d)[M];
DATA_TYPE *mean_d;
DATA_TYPE *stddev_d;

gpuErrchk(cudaMalloc(&data_d, n * m * sizeof(DATA_TYPE)));
gpuErrchk(cudaMalloc(&symmat_d, m * m * sizeof(DATA_TYPE)));
gpuErrchk(cudaMalloc(&mean_d, m * sizeof(DATA_TYPE)));
gpuErrchk(cudaMalloc(&stddev_d, m * sizeof(DATA_TYPE)));

/* Pinned host memory */
DATA_TYPE (*data)[N];
DATA_TYPE (*symmat)[M];

gpuErrchk(cudaMallocHost(&data, n * m * sizeof(DATA_TYPE)));
gpuErrchk(cudaMallocHost(&symmat, m * m * sizeof(DATA_TYPE)));

DATA_TYPE (*data_host)[N];
DATA_TYPE (*symmat_host)[M];
DATA_TYPE *mean_host;
DATA_TYPE *stddev_host;

data_host = (DATA_TYPE (*)[N])
    malloc(n * m * sizeof(DATA_TYPE));

symmat_host = (DATA_TYPE (*)[M])
    malloc(m * m * sizeof(DATA_TYPE));

mean_host = (DATA_TYPE *)
    malloc(m * sizeof(DATA_TYPE));

stddev_host = (DATA_TYPE *)
    malloc(m * sizeof(DATA_TYPE));

if (!data_host || !symmat_host ||
    !mean_host || !stddev_host)
{
    fprintf(stderr, "Host memory allocation failed\n");
    exit(EXIT_FAILURE);
}    
  /* Initialize array(s). */
  
 init_array(m, n, &float_n, data);
 init_array(m, n, &float_n, data_host);
  
/* Measure GPU execution time. */
// auto gpu_start = std::chrono::steady_clock::now();

// kernel_correlation_gpu(m, n, float_n,data, symmat, mean, stddev);

// auto gpu_end = std::chrono::steady_clock::now();

auto gpu_start = std::chrono::steady_clock::now();

/* Transfer input from pinned host memory to device memory. */
gpuErrchk(cudaMemcpy(data_d, data,
                     n * m * sizeof(DATA_TYPE),
                     cudaMemcpyHostToDevice));

/* Execute the four CUDA kernels. */
kernel_correlation_gpu(m, n, float_n,
                       data_d, symmat_d, mean_d, stddev_d);

/* Transfer results from device memory to pinned host memory. */
gpuErrchk(cudaMemcpy(symmat, symmat_d,
                     m * m * sizeof(DATA_TYPE),
                     cudaMemcpyDeviceToHost));

auto gpu_end = std::chrono::steady_clock::now();


double gpu_time = std::chrono::duration<double>(gpu_end - gpu_start).count();

/* Measure host execution time for comparison. */
auto host_start = std::chrono::steady_clock::now();

kernel_correlation_host(m, n, float_n,data_host, symmat_host,mean_host, stddev_host);

auto host_end = std::chrono::steady_clock::now();

double host_time = std::chrono::duration<double>(host_end - host_start).count();
/* Calculate achieved GFLOPS. */
double flops = (double)n * m * (m - 1);

double gpu_gflops = flops / (gpu_time * 1e9);

double host_gflops = flops / (host_time * 1e9);
/* Report execution results. */
printf("correlation (GPU):  %10.6f sec  %8.3f GFLOPS\n",gpu_time, gpu_gflops);

printf("correlation (HOST): %10.6f sec  %8.3f GFLOPS\n",host_time, host_gflops);

/* Validate GPU output against the host implementation. */
double max_error = 0.0;
double tolerance = 1e-5;
bool valid = true;

for (int i = 0; i < m; i++)
{
    for (int j = 0; j < m; j++)
    {
        double error = fabs(
            (double)symmat[i][j] - symmat_host[i][j]
        );

        if (!isfinite(error))
        {
            valid = false;
            continue;
        }

        if (error > max_error)
            max_error = error;

        if (error > tolerance)
            valid = false;
    }
}

printf("Maximum error: %.10e\n", max_error);

printf("Validation: %s\n",valid ? "PASSED" : "FAILED");
  
  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
 polybench_prevent_dce(print_array(m, symmat));

/* Free managed and host memory. */
// 
/* Free device memory. */
gpuErrchk(cudaFree(data_d));
gpuErrchk(cudaFree(symmat_d));
gpuErrchk(cudaFree(mean_d));
gpuErrchk(cudaFree(stddev_d));

/* Free pinned host memory. */
gpuErrchk(cudaFreeHost(data));
gpuErrchk(cudaFreeHost(symmat));

free(data_host);
free(symmat_host);
free(mean_host);
free(stddev_host);

  return 0;
}


/** 
Kernel              	Grid type	     Work per thread       	Complexity:

Mean	             ==>  1D	         One column	            O(NM)
Standard deviation ==>	1D	         One column	            O(NM)
Normalization      ==>	2D	         One element	          O(NM)
Correlation	       ==>  2D	         One pair of columns	  O(NM²)

 */
