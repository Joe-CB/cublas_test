
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <stdio.h>
#include <vector>

cuComplex multy(cuComplex&a, cuComplex&b)
{
	cuComplex c;
	c.x = a.x*b.x - a.y*b.y;
	c.y = a.x*b.y + a.y*b.x;
	return c;
}


int main()
{
	const int m = 1000;
	const int n = 1600;
	const int k = 1000;
	cuComplex *dm_A, *dm_B, *d_C;
	std::vector<cuComplex> C(m*n);
	std::vector<cuComplex> h_c(m*n);
	//Malloc
	cudaMallocManaged((void**)&dm_A, sizeof(cuComplex)*m*k);
	cudaMallocManaged((void**)&dm_B, sizeof(cuComplex)*k*n);
	cudaMalloc((void**)&d_C, sizeof(cuComplex)*m*n);
	//Init data
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < k; j++)
		{
			//行优先
			dm_A[i*k + j].x = i*j;
			dm_A[i*k + j].y = j;
		}
	}
	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < n; j++)
		{
			//行优先
			dm_B[i*n + j].x = j;
			dm_B[i*n + j].y = i*j;
		}
	}
	//cublas
	cublasHandle_t handle;
	cublasCreate(&handle);

	//Ain Arow是行优先的
	cuComplex alpha, beta;
	alpha.x = 1; alpha.y = 0;
	beta.x = 0; beta.y = 0;
	cublasCgemm_v2(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		n, m, k,
		&alpha,
		dm_B, n,
		dm_A, k,
		&beta,
		d_C, n);
	cudaDeviceSynchronize();
	cudaMemcpy(&(h_c[0]), d_C, sizeof(cuComplex)*m*n, cudaMemcpyDeviceToHost);
	cublasDestroy(handle);

	//cpu
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cuComplex temp;
			temp.x = 0; temp.y = 0;
			for (int t = 0; t < k; t++)
			{
				cuComplex b = multy(dm_A[i*k + t], dm_B[t*n + j]);
				temp.x += b.x;
				temp.y += b.y;
			}
			C[i*n + j] = temp;
		}
	}
	//Verify
	system("pause");
}
