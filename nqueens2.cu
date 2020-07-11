#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define N 8
#define N2 N*N
#define DEPTH 6

typedef struct Cartesian Cartesian;

struct Cartesian {
	int x;
	int y;
};

__device__ void display_frontier(Cartesian frontier[N2], int size) {
	int i;
	printf("\nFrontier:\n");
	for (i = 0; i < size; i++) {
		printf("[%d, %d] ", frontier[i].x, frontier[i].y);
	}
	printf("---------------------------------------------");
}

__device__ void display_board(int board[N][N]) {
	int i, j;

	printf("\n.");
	for (i = 0; i < N; i++) {
		printf("____");
	}
	printf(".\n");
	for (i = 0; i < N; i++) {
		printf("|");
		for (j = 0; j < N; j++) {
			printf("_%d_|", board[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}




__device__ int is_attacked(int board[N][N], int x, int y);

__device__ int solve(int size, int board[N][N], const int numThreads);

__global__ void nQueenKernel(int* solutions, int size, const int numThreads) {
	int board[N][N];
	int i, j;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			board[i][j] = 0;
		}
	}
	
	solutions[blockIdx.x * numThreads + threadIdx.x] = solve(size, board, numThreads);
}

int main() {
	
	//int i, j;
	int k;
	int cont;
	
	int* dev_solutions = NULL;
	int numThreads = (int)pow(double(N), double(DEPTH));
	int numBlocks = 1;

	int* h_solutions;
	cudaError_t cudaStatus;
	float start = (float)clock() / CLOCKS_PER_SEC;
	float end;


	while (numThreads > 1024) {
		numThreads = numThreads / N; // ensured to always be divisible since numthreads is a power of N
		numBlocks = numBlocks * N;
	}
	printf("NumThreadsPerBlock = %d, NumBlocks = %d, Size = %d\n", numThreads, numBlocks, numThreads*numBlocks);
	h_solutions = (int*)calloc(numThreads * numBlocks, sizeof(int));
	if (!h_solutions) {
		printf("EXCEPTION: calloc failed for h_solutions!\n");
		return 0;
	}


	// empty board
	/*
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			board[i][j] = 0;
		}
	}*/

	// solve and print number of solutions for a x paramater 

	// SEQUENTIAL 
	/*
	for (i = 0; i < N; i++) {
		printf("%d\n", i);
		cont += solve(i, board);
	}
	*/

	// PARALLEL
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed! Are you sure you have a CUDA-capable GPU?");
		return 0;
	}
	cudaStatus = cudaMalloc((void**)& dev_solutions, numThreads * numBlocks * sizeof(int)); 
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return 0;
	}
	
	cudaStatus = cudaMemcpy(dev_solutions, h_solutions, numThreads * numBlocks * sizeof(int), cudaMemcpyHostToDevice);
	nQueenKernel << <numBlocks, numThreads >> > ((int*)dev_solutions, numBlocks * numThreads, numThreads);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "nQueneKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return 0;
	}

	cudaStatus = cudaMemcpy(h_solutions, dev_solutions, numThreads * numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

	cont = 0;
	for (k = 0; k < numBlocks * numThreads; k++) {
		cont += h_solutions[k];
	}
	end = (float)clock() / CLOCKS_PER_SEC;
	printf("\nSolutions = %d. Time spent: %f\n", cont, end - start);
	cudaFree(dev_solutions);
	free(h_solutions);
	cudaDeviceReset();
	return 0;
}
__device__ int is_attacked(int board[N][N], int x, int y) {
	int i, j;
	int attacked = 0;
	if (y == 0)
		return attacked;

	for (i = 0; i < N && !attacked; i++) {
		for (j = 0; j < y && !attacked; j++) {
			if (board[i][j])
				if (i == x || i + j == x + y || i - j == x - y)
					attacked = 1;
		}
	}
	return attacked;
}

__device__ int solve(int size, int board[N][N], const int numThreads) {
	Cartesian frontier[N2];
	int i, j;
	int k, depth;
	int d;
	int previous_depth;
	unsigned int cont;
	unsigned int solution_count;
	int x;
	int div;

	solution_count = 0;

	if (size == 1) {
		for (i = 0; i < N; i++) {
			frontier[i].x = i;
			frontier[i].y = 0;
		}
		cont = i;
		previous_depth = -1;
		goto SearchStart;
	}
	if (size % N != 0) {
		printf("EXCEPTION: invalid thread size");
		return 0;
	}
	for (i = 0; i < (N * N); i++) {
		frontier[i].x = 0;
		frontier[i].y = 0;
	}
	x = blockIdx.x * numThreads + threadIdx.x;
	depth = __float2int_rn(log10f((float)size) / log10f((float)N)) - 1;
	for (d = 0; d < depth; d++) {
		div = 1;
		for (i = 0; i < (depth - d); i++) {
			div = div * N;
		}
		k = x / div;
		if (is_attacked(board, k, d)) {
			return 0; // useless branch
		}
		if (k > N || d > N) {
			printf("EXCEPTION: math error! k = %d, d = %d\n at [b,t] = [%d, %d]", k, d, blockIdx.x, threadIdx.x);
			return 0;
		}
		board[k][d] = 1;
		x = __float2int_rd(fmodf(__int2float_rn(x), __float2int_rn(powf((float)N, (float)(depth - d)))));
	}
	if (is_attacked(board, x % N, depth)) {
		return 0; // useless branch
	}
	frontier[0].x = x % N;
	frontier[0].y = depth;
	cont = 1;
	previous_depth = depth - 1;
SearchStart:
	while (cont > 0) {
		/*
		printf("\ndepth = %d\n", depth);
		printf("\nblock %d ", blockIdx.x);
		display_frontier(frontier, cont);
		*/
		// pop last element of frontier (depth-first search)
		k = frontier[cont - 1].x;
		depth = frontier[cont - 1].y;
		frontier[cont - 1].x = 0;
		frontier[cont - 1].y = 0;
		cont -= 1;

		//printf("\ndepth = %d\n", depth);
		if (depth == N - 1) {
			// found solution

			board[k][depth] = 1;
			//display_board(board);
			solution_count++;
			previous_depth = depth;
			continue;
		}

		if (previous_depth >= depth) {
			// backtrack
			for (i = 0; i < N; i++) {
				for (j = previous_depth; j >= depth; j--)
					board[i][j] = 0;
			}
		}
		board[k][depth] = 1;
		previous_depth = depth;
		depth += 1;
		/*printf("\ndepth = %d\n", depth);
		display_board(board);
		*/
		// investigate all 0 < t < N elements of row = depth for candidacy
		for (k = 0; k < N; k++) {
			if (is_attacked(board, k, depth))
				continue;
			frontier[cont].x = k;
			frontier[cont].y = depth;
			cont++;
		}
	}
	//printf("whoops!");

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			board[i][j] = 0;
		}
	}

	//printf("\n%d", blockIdx.x);

	return solution_count;
}


