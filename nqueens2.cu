

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

/*Code by gabrielesvb
 *Cuda error handling taken from CUDA VS2019 sample project
 * Run instructions:
 *  - Have Cuda Toolkit installed (developed with Cuda Toolkit v11.0)
 *  - Have a Cuda-capable GPU
 *  - Define desired parameters N and DEPTH at compile time
 *  - Simply build and run in a Visual Studio CUDA solution
 */ 
#define N 8
#define N2 N*N
#define DEPTH 6

typedef struct Cartesian Cartesian;

struct Cartesian {
	int x;
	int y;
};

// Display methods aren't useful in parallel execution since concurrence is a problem for printing to stdout
/*
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
*/



__device__ int is_attacked(int board[N][N], int x, int y);

__device__ int solve(int size, int board[N][N], const int numThreads);

// Main CUDA Kernel, responsible for calling solve() function for each thread and assigning the result to the respective thread element of solutions vector
__global__ void nQueenKernel(int* solutions, int size, const int numThreads) {
	int board[N][N];
	int i, j;

	// initialize board to all zeroes
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			board[i][j] = 0;
		}
	}
	
	solutions[blockIdx.x * numThreads + threadIdx.x] = solve(size, board, numThreads);
}

// Driver function for calculating the number of threads and blocks from N,
// allocating memory in the GPU, running the CUDA Kernel, and finally printing
// out the result of the test (both number of solutions and time spent)
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

	// Calculate number of blocks and threads from N 
	while (numThreads > 1024) {
		numThreads = numThreads / N; // ensured to always be divisible since numthreads is a power of N
		numBlocks = numBlocks * N;
	}
	printf("NumThreadsPerBlock = %d, NumBlocks = %d, Size = %d\n", numThreads, numBlocks, numThreads*numBlocks);

	// Allocate host memory
	h_solutions = (int*)calloc(numThreads * numBlocks, sizeof(int));
	if (!h_solutions) {
		printf("EXCEPTION: calloc failed for h_solutions!\n");
		return 0;
	}

	// Sets CUDA-capable GPU device
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed! Are you sure you have a CUDA-capable GPU?");
		return 0;
	}

	// Allocate device memory
	cudaStatus = cudaMalloc((void**)& dev_solutions, numThreads * numBlocks * sizeof(int)); 
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return 0;
	}
	
	// Copies host data (all zeroes from calloc) to device
	cudaStatus = cudaMemcpy(dev_solutions, h_solutions, numThreads * numBlocks * sizeof(int), cudaMemcpyHostToDevice);

	// Launch Kernel
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

	// Copies device results vector into host vector 
	cudaStatus = cudaMemcpy(h_solutions, dev_solutions, numThreads * numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

	// Sum all elements in host vector to obtain solution
	cont = 0;
	for (k = 0; k < numBlocks * numThreads; k++) {
		cont += h_solutions[k];
	}

	// end clock
	end = (float)clock() / CLOCKS_PER_SEC;

	// print results
	printf("\nSolutions = %d. Time spent: %f\n", cont, end - start);

	// free everything
	cudaFree(dev_solutions);
	free(h_solutions);
	cudaDeviceReset();
	return 0;
}

/* Function to verify if a given (x, y) square is currently attacked in board
 * Returns true if so, false otherwise
 */
__device__ int is_attacked(int board[N][N], int x, int y) {
	int i, j;
	int attacked = 0;
	if (y == 0)
		return attacked;

	// Since the algorithm in solve() ensures there is no rank conflict, the only
	// conflicts to check are file and diagonal. File conflict means x1 = x2, whilst
	// diagonal conflict means either the sum of the coordinates or the subtraction
	// of the 2D coordinates are the same
	for (i = 0; i < N && !attacked; i++) {
		for (j = 0; j < y && !attacked; j++) {
			if (board[i][j])
				if (i == x || i + j == x + y || i - j == x - y)
					attacked = 1;
		}
	}
	return attacked;
}


/* Solve function for the NQueen problem using CUDA. 
 * Receives parameters:
 * @ size = total number of threads (numThreads*numBlocks)
 * @ board = 2D board (ASSUMED TO BE ALL ZEROES)
 * @ numThreads = number of threads per block
 */ 
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

	// If size is 1 then this is a trivial sequential solution being executed by a single thread (good luck!)
	if (size == 1) {
		for (i = 0; i < N; i++) {
			frontier[i].x = i;
			frontier[i].y = 0;
		}
		cont = i;
		previous_depth = -1;
		goto SearchStart;
	}

	// Assert thread size logic
	if (size % N != 0) {
		printf("EXCEPTION: invalid thread size");
		return 0;
	}

	// Initialise frontier vector to all zeroes
	for (i = 0; i < (N * N); i++) {
		frontier[i].x = 0;
		frontier[i].y = 0;
	}

	// 1st transformation: linearizing 2D coordinates
	x = blockIdx.x * numThreads + threadIdx.x;

	// Could otherwise be obtained from DEPTH directly, but it does assert 
	// that the number of roots be equal to the number of threads.
	depth = __float2int_rn(log10f((float)size) / log10f((float)N)) - 1;

	// From DEPTH and N, infer the creation of all roots. 
	// All threads will run the search starting from a single unique root. 
	// We start from the top of the board, going down until we reach DEPTH.
	// depth = 0 means we start from all possible Queen positions in the first
	// rank as our roots. For depth > 0 we build our roots from the branches of 
	// previous iterations
	for (d = 0; d < depth; d++) {
		// For each rank, 
		div = 1;

		// Get the dividend for the 2nd transformation (dividend should shrink
		// iteratively as a factor of N)
		for (i = 0; i < (depth - d); i++) {
			div = div * N;
		}

		// Calculate the position of the piece in this thread from the integer
		// division between a linearized thread id and div 
		k = x / div;
		
		// If this thread is trying to place a queen where it is already attacked, return no solutions
		if (is_attacked(board, k, d)) {
			return 0; // useless branch
		}

		// Mere assertion that math works
		if (k > N || d > N) {
			printf("EXCEPTION: math error! k = %d, d = %d\n at [b,t] = [%d, %d]", k, d, blockIdx.x, threadIdx.x);
			return 0;
		}

		// update board
		board[k][d] = 1;

		// update thread id from previous id mod pow(N, depth-d)
		x = __float2int_rd(fmodf(__int2float_rn(x), __float2int_rn(powf((float)N, (float)(depth - d)))));

		// ... continues until we reach desired depth
	}

	// Same as before, if attacked at desired depth, then quit search and return 0
	if (is_attacked(board, x % N, depth)) {
		return 0; // useless branch
	}

	// start search by adding the relevant position to this thread at the desired DEPTH to the frontier
	frontier[0].x = x % N;
	frontier[0].y = depth;

	// Search proceeds just like sequential code
	cont = 1;
	previous_depth = depth - 1;

SearchStart:

	// Will search from desired depth until the last rank of the board
	while (cont > 0) {

		// pop last element of frontier (depth-first search)
		k = frontier[cont - 1].x;
		depth = frontier[cont - 1].y;
		frontier[cont - 1].x = 0;
		frontier[cont - 1].y = 0;
		cont -= 1;

		// If depth is the last rank,
		if (depth == N - 1) {

			// found solution
			board[k][depth] = 1;
			//display_board(board);
			solution_count++;
			previous_depth = depth;
			continue;
		}

		// If we detect we are backtracking from the last step, update board accordingly
		if (previous_depth >= depth) {
			// backtrack
			for (i = 0; i < N; i++) {
				for (j = previous_depth; j >= depth; j--)
					board[i][j] = 0;
			}
		}

		// update board and depth values
		board[k][depth] = 1;
		previous_depth = depth;
		depth += 1;

		// investigate all 0 < t < N elements of row = depth for candidacy
		for (k = 0; k < N; k++) {
			if (is_attacked(board, k, depth))
				continue;
			frontier[cont].x = k;
			frontier[cont].y = depth;
			cont++;
		}
	}

	// Just zeroing out board before exiting
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			board[i][j] = 0;
		}
	}

	//printf("\n%d", blockIdx.x);

	return solution_count;
}


