#include <stdio.h>
#define N 6
#define N2 N*N


typedef struct Cartesian Cartesian;

struct Cartesian {
	int x;
	int y;
};

void display_frontier(Cartesian frontier[N2], int size) {
	int i;
	printf("\nFrontier:\n");
	for (i = 0; i < size; i++) {
		printf("[%d, %d] ", frontier[i].x, frontier[i].y);
	}
	printf("---------------------------------------------");
}

void display_board(int board[N][N]) {
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

int solve(int x, int board[N][N]);

int main() {
	int board[N][N];
	int i, j;
	int cont = 0;

	// empty board
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			board[i][j] = 0;
		}
	}

	// solve and print number of solutions for a x paramater
	for (i = 0; i < N; i++)
		cont += solve(i, board);

	printf("No.Solucoes = %d", cont);
	return 0;
}

int solve(int x, int board[N][N]) {
	Cartesian frontier[N2];
	int i, j;
	int k;

	int depth = 0;
	int previous_depth = -1;
	int skip_k;
	unsigned int cont;
	unsigned int solution_count = 0;

	for (i = 0; i < (N ^ 2); i++) {
		frontier[i].x = 0;
		frontier[i].y = 0;
	}

	frontier[0].x = x;
	frontier[0].y = 0;
	cont = 1;




	while (cont > 0) {
		display_board(board);
		display_frontier(frontier, cont);
		/*
		printf("\ndepth = %d\n", depth);
		*/
		// pop last element of frontier (depth-first search)
		k = frontier[cont - 1].x;
		depth = frontier[cont - 1].y;
		frontier[cont - 1].x = 0;
		frontier[cont - 1].y = 0;
		cont -= 1;

		display_frontier(frontier, cont);
		//printf("\ndepth = %d\n", depth);
		if (depth == N - 1) {
			// found solution

			board[k][depth] = 1;
			display_board(board);
			solution_count++;
			previous_depth = depth;
			continue;
		}

		if (previous_depth >= depth) {
			// backtrack
			printf("Hey Jude");
			for (i = 0; i < N; i++) {
				for (j = previous_depth; j >= depth; j--)
					board[i][j] = 0;
			}
		}
		board[k][depth] = 1;
		previous_depth = depth;
		depth += 1;
		printf("\n\nprev%d cur%d\n\n", previous_depth, depth);
		/*printf("\ndepth = %d\n", depth);
		*/
		display_board(board);
		// investigate all 0 < t < N elements of row = depth for candidacy
		for (k = 0; k < N; k++) {
			skip_k = 0;
			printf("%d\n\n", depth);
			for (i = 0; i < N && !skip_k; i++) {
				for (j = 0; j < depth && !skip_k; j++) {

					// File or Diagonal conflict  (ps: rank conflict never happens because of increasing depth)

					if (board[i][j]) {
						if (i == k || (i + j == k + depth) || (i - j == k - depth)) {

							skip_k = 1;
						}
					}

				}
			}

			if (!skip_k) {
				printf("k = %d, depth = %d\n", k, depth);
				frontier[cont].x = k;
				printf("%d\n\n", depth);
				frontier[cont].y = depth;
				cont++;
			}
		}

		printf("\nabutre\n");
		display_frontier(frontier, cont);
	}
	printf("whoops!");

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			board[i][j] = 0;
		}
	}

	return solution_count;
}