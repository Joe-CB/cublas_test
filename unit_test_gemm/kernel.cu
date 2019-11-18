
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include	<vector>
#include <stdio.h>

using Complex = float2;

#define CUBE_SIZE 16
#define TILE_WIDTH 16
void get_last_error()
{
	cudaError_t cudastatus = cudaGetLastError();
	if (cudastatus != cudaSuccess)
	{
		printf("%s", cudaGetErrorString(cudastatus));
	}
}

namespace global {
	namespace device {
		// Complex addition
		static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
			Complex c;
			c.x = a.x + b.x;
			c.y = a.y + b.y;
			return c;
		}

		// Complex minus
		static __device__ __host__ inline Complex ComplexMinus(Complex a, Complex b) {
			Complex c;
			c.x = a.x - b.x;
			c.y = a.y - b.y;
			return c;
		}

		// Complex scale
		static __device__ __host__ inline Complex ComplexScale(Complex a, float s) {
			Complex c;
			c.x = s * a.x;
			c.y = s * a.y;
			return c;
		}

		// Complex multiplication
		static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
			Complex c;
			c.x = a.x * b.x - a.y * b.y;
			c.y = a.x * b.y + a.y * b.x;
			return c;
		}
	}//device
	__global__ void complex_gemm(Complex* A, Complex* B, int m, int k, int n, Complex* C)
	{
		__shared__	Complex	s_a[CUBE_SIZE][CUBE_SIZE];
		__shared__	Complex	s_b[CUBE_SIZE][CUBE_SIZE];
		//当前处理的块为：
		int block_row = blockIdx.x;
		int block_col = blockIdx.y;
		int thread_row = threadIdx.x;
		int thread_col = threadIdx.y;

		int global_row = blockDim.x*blockIdx.x;
		int global_col = blockDim.y*blockIdx.y;
		__shared__ int times;
		if (thread_row == 0 && thread_col == 0)
		{
			times = k / CUBE_SIZE;
			times += k % CUBE_SIZE ? 1 : 0;
		}
		__syncthreads();
		//载入数据
		Complex sum;
		sum.x = 0; sum.y = 0;
		for (int i = 0; i < times;i++)
		{
			//对于A，读入第global_row + thread_row行，第i*CUBE_SIZE + thread_col列的数据
			if (global_row + thread_row >= m || i*CUBE_SIZE + thread_col >= k)
			{
				s_a[thread_row][thread_col].x = 0;
				s_a[thread_row][thread_col].y = 0;
			}
			else
				s_a[thread_row][thread_col] = A[(global_row + thread_row)*k + i*CUBE_SIZE + thread_col];
			//对于B，读入第[i*CUBE_SIZE + thread_row]行，第[global_col + thread_col]列的数据
			if (i*CUBE_SIZE + thread_row >= k || global_col + thread_col >= n)
			{
				s_b[thread_row][thread_col].x = 0;
				s_b[thread_row][thread_col].y = 0;
			}
			else
				s_b[thread_row][thread_col] = B[(i*CUBE_SIZE + thread_row)*n + global_col + thread_col];				//x固定，y移动
			__syncthreads();
			for (int j = 0; j < CUBE_SIZE; j++)
			{
				Complex temp = device::ComplexMul(s_a[thread_row][j], s_b[j][thread_col]);
				sum = device::ComplexAdd(temp,sum);
			}
		}
		C[(global_row + thread_row)*n + global_col + thread_col] = sum;
	}

	__global__ void complex_gemm_shared(Complex* A, Complex* B, int m, int k, int n, Complex* C)
	{
		__shared__ Complex s_a[TILE_WIDTH][TILE_WIDTH];
		__shared__ Complex s_b[TILE_WIDTH][TILE_WIDTH];

		int t_row = threadIdx.y;		//通常x表示列，y表示行，C++中行优先，因此x时连续的
		int t_col = threadIdx.x;
		int Row = blockIdx.y*blockDim.y + t_row;
		int Col = blockIdx.x*blockDim.x + t_col;
		if (Col == 992)
		{
			Col = 992;
			int k = 10;
		}
		float2 sum;
		sum.x = 0; sum.y = 0;
		for (int t = 0; t < (k - 1) / TILE_WIDTH + 1; ++t)
		{
			//读入共享内存
			if (Row < m&&t*TILE_WIDTH + t_col < k)	//A是否越界
			{
				s_a[t_row][t_col] = A[Row*k + t*TILE_WIDTH + t_col];
			}
			else
			{
				s_a[t_row][t_col].x = 0;
				s_a[t_row][t_col].y = 0;
			}
			if (t*TILE_WIDTH + t_row < k&&Col < n)
			{
				s_b[t_row][t_col] = B[(t*TILE_WIDTH + t_row)*n + Col];
			}
			else
			{
				s_b[t_row][t_col].x = 0;
				s_b[t_row][t_col].y = 0;
			}
			__syncthreads();
//#pragma unroll
			for (int j = 0; j < TILE_WIDTH; j++)
			{
				sum.x += (s_a[t_row][j].x*s_b[j][t_col].x - s_a[t_row][j].y*s_b[j][t_col].y);
				sum.y += (s_a[t_row][j].x*s_b[j][t_col].y + s_a[t_row][j].y*s_b[j][t_col].x);
				//Complex temp = device::ComplexMul(s_a[t_row][j], s_b[j][t_col]);
				//sum = device::ComplexAdd(sum, temp);
			}
			__syncthreads();
		}
	
		if (Row < m&&Col < n)
		{
			C[Row*n + Col] = sum;
			//C[Row*n + Col].x = sum.x;
			//C[Row*n + Col].y = sum.y;
		}
			
	}
	//加了求和矫正后的
	__global__ void complex_gemm_shared_correction(Complex* A, Complex* B, int m, int k, int n, Complex* C)
	{
		__shared__ Complex s_a[TILE_WIDTH][TILE_WIDTH];
		__shared__ Complex s_b[TILE_WIDTH][TILE_WIDTH];

		int t_row = threadIdx.y;		//通常x表示列，y表示行，C++中行优先，因此x时连续的
		int t_col = threadIdx.x;
		int Row = blockIdx.y*blockDim.y + t_row;
		int Col = blockIdx.x*blockDim.x + t_col;
		if (Col == 992)
		{
			Col = 992;
			int k = 10;
		}
		Complex sum, deviation;
		deviation.x = 0; deviation.y = 0;
		sum.x = 0; sum.y = 0;
		for (int t = 0; t < (k - 1) / TILE_WIDTH + 1; ++t)
		{
			//读入共享内存
			if (Row < m&&t*TILE_WIDTH + t_col < k)	//A是否越界
			{
				s_a[t_row][t_col] = A[Row*k + t*TILE_WIDTH + t_col];
			}
			else
			{
				s_a[t_row][t_col].x = 0;
				s_a[t_row][t_col].y = 0;
			}
			if (t*TILE_WIDTH + t_row < k&&Col < n)
			{
				s_b[t_row][t_col] = B[(t*TILE_WIDTH + t_row)*n + Col];
			}
			else
			{
				s_b[t_row][t_col].x = 0;
				s_b[t_row][t_col].y = 0;
			}
			__syncthreads();
			for (int j = 0; j < TILE_WIDTH; j++)
			{
				//sum.x += (s_a[t_row][j].x*s_b[j][t_col].x - s_a[t_row][j].y*s_b[j][t_col].y);
				//sum.y += (s_a[t_row][j].x*s_b[j][t_col].y + s_a[t_row][j].y*s_b[j][t_col].x);
				Complex temp = device::ComplexMul(s_a[t_row][j], s_b[j][t_col]);
				temp = device::ComplexMinus(temp, deviation);
				Complex added = device::ComplexAdd(sum, temp);
				deviation = device::ComplexMinus(device::ComplexMinus(added, sum), temp);
				sum = device::ComplexAdd(sum, temp);
			}
			__syncthreads();
		}

		if (Row < m&&Col < n)
			C[Row*n + Col] = sum;
	}

	__global__ void complex_gemm_no_shared(Complex* A, Complex* B, int m, int k, int n, Complex* C)
	{
		//当前处理的块为：
		int global_row = blockDim.y*blockIdx.y + threadIdx.y;
		int global_col = blockDim.x*blockIdx.x + threadIdx.x;

		if (global_col < n && global_row < m)
		{
			Complex sum;
			sum.x = 0; sum.y = 0;
			for (int i = 0; i < k; i++)
			{
				Complex temp = device::ComplexMul(A[global_row*k + i], B[i*n + global_col]);
				sum = device::ComplexAdd(temp, sum);
			}
			C[global_row*n + global_col] = sum;
		}
	}

}

void gemm_cpu(Complex *A, Complex* B, int m, int k, int n, Complex* C)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			Complex sum;
			sum.x = sum.y = 0;
			for (int idx = 0; idx < k; idx++) {
				Complex temp = global::device::ComplexMul(A[i*k+idx], B[idx*n+j]);
				sum = global::device::ComplexAdd(sum, temp);
			}
			C[i*n + j] = sum;
		}
	}
		
}

int main()
{
	const int m = 1000;
	const int k = 1600;
	const int n = 1000;

	//malloc Data
	Complex* m_A;
	Complex* m_B;
	Complex* m_C;
	Complex* m_C2;
	std::vector<Complex> h_C;
	cudaMallocManaged((void**)&m_A, sizeof(Complex)*m*k);
	cudaMallocManaged((void**)&m_B, sizeof(Complex)*n*k);
	cudaMallocManaged((void**)&m_C, sizeof(Complex)*m*n);
	cudaMallocManaged((void**)&m_C2, sizeof(Complex)*m*n);
	h_C.resize(m*n);
	get_last_error();
	//set Data
	for (int i = 0; i < m*k; i++)
	{
		m_A[i].x = rand()/float(RAND_MAX)*100;
		m_A[i].y = rand() / float(RAND_MAX) * 100;
	}
	for (int i = 0; i < n*k; i++)
	{
		m_B[i].x = rand() / float(RAND_MAX) * 100;
		m_B[i].y = rand() / float(RAND_MAX) * 100;
	}

	//CPU gemm
	//gemm_cpu(m_A, m_B, m, k, n, &(h_C[0]));
	//GPU gemm

	dim3 block_size;
	block_size.x = CUBE_SIZE;
	block_size.y = CUBE_SIZE;
	dim3 grid_size;
	grid_size.x = m / CUBE_SIZE;
	grid_size.x += m%CUBE_SIZE ? 1 : 0;
	grid_size.y = n / CUBE_SIZE;
	grid_size.y += n%CUBE_SIZE ? 1 : 0;

	//global::complex_gemm_no_shared <<<grid_size, block_size >> >(m_A, m_B, m, k, n, m_C);
	cudaDeviceSynchronize();
	get_last_error();
	global::complex_gemm_shared <<<grid_size, block_size>>>(m_A, m_B, m, k, n, m_C2);
	get_last_error();
	cudaDeviceSynchronize();

	return 0;
	Complex* C =  &(h_C[0]);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			int idx = i*n + j;
			if (C[idx].x != m_C2[idx].x)
			{
				printf("x:%d-%d__%f::%f.\n", i,j, C[idx].x, m_C2[idx].x);
				printf("y:%d-%d__%f::%f.\n", i, j, C[idx].y, m_C2[idx].y);
			}
			if (C[idx].y != m_C2[idx].y)
			{
				printf("y:%d-%d__%f::%f.\n", i, j, C[idx].y, m_C2[idx].y);
			}
		}
	}
		
	system("pause");
	return 0;
}
