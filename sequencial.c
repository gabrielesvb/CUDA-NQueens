#include <stdio.h>
#include <time.h>

#define N 4
#define N2 N*N


typedef struct Cartesian Cartesian;

struct Cartesian{
    int x;
    int y;
};

void display_frontier(Cartesian frontier[N2], int size){
    int i;
    printf("\nFrontier:\n");
    for (i = 0; i < size; i++){
        printf("[%d, %d] ", frontier[i].x, frontier[i].y);
    }
    printf("---------------------------------------------");
}

void display_board(int board[N][N]){
    int i, j;

    printf("\n.");
    for (i = 0; i < N; i++){
        printf("____");
    }
    printf(".\n");
    for (i = 0; i < N; i++){
        printf("|");
        for (j=0; j<N; j++){
            printf("_%d_|", board[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int solve(int x, int board[N][N]);

int main(){
    int board[N][N];
    int i, j;
    int cont = 0;
    float start = (float)clock()/CLOCKS_PER_SEC;
    float end;

    // empty board
    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            board[i][j] = 0;
        }
    }

    // solve and print number of solutions for a x paramater
    for (i = 0; i < N; i++){
        printf("%d\n", i);
        cont += solve(i, board);
    }

    end = (float)clock()/CLOCKS_PER_SEC;
    printf("No.Solucoes = %d. Time spent = %.5f", cont, end);
    return 0;
}

int solve(int x, int board[N][N]){
    Cartesian frontier[N2];
    int i, j;
    int k;
    int depth = 0;
    int previous_depth = -1;
    int skip_k;
    unsigned int cont;
    unsigned int solution_count = 0;
    for (i = 0; i < (N^2); i++){
        frontier[i].x = 0;
        frontier[i].y = 0;
    }
    frontier[0].x = x;
    frontier[0].y = 0;
    cont = 1;
    while (cont > 0){
        /*
        printf("\ndepth = %d\n", depth);
        display_frontier(frontier, cont);
        */
        // pop last element of frontier (depth-first search)
        k = frontier[cont - 1].x;
        depth = frontier[cont - 1].y;
        frontier[cont - 1].x = 0;
        frontier[cont - 1].y = 0;
        cont -= 1;
        //printf("\ndepth = %d\n", depth);
        if (depth == N - 1){
            // found solution
            board[k][depth] = 1;
            //display_board(board);
            solution_count++;
            previous_depth = depth;
            continue;
        }
        if (previous_depth >= depth){
            // backtrack
            for(i = 0; i < N; i++){
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
        for (k = 0; k < N; k++){
            skip_k = 0;
            for (i = 0; i < N && !skip_k; i++){
                for (j = 0; j < depth && !skip_k; j++){

                    // File or Diagonal conflict  (ps: rank conflict never happens because of increasing depth)
                    
                    if(board[i][j]){
                        if (i == k || (i + j == k + depth) || (i - j == k - depth)){
                            
                            skip_k = 1;
                        }
                    }

                }
            }
            if (!skip_k){
            
                frontier[cont].x = k;
                frontier[cont].y = depth;
                cont++;
            }
        }   
    }
    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            board[i][j] = 0;
        }
    }

    return solution_count;
}