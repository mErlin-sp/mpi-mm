/*FILE: mpi_mm_blocking.c
* by Blaise Barney
* modified by Oleksandr Popov
*/

/* Run with mpirun --use-hwthread-cpus -n 4 mpi_mm */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NRA 10          /* number of rows in matrix A */
#define NCA 10          /* number of columns in matrix A */
#define NCB 10          /* number of columns in matrix B */
#define MASTER 0        /* taskID of first task */
#define FROM_MASTER 1   /* setting a message type */
#define FROM_WORKER 2   /* setting a message type */

void print_matrix(double *m, int nr, int nc, char *msg);

int main(int argc, char *argv[]) {
    printf("Start\n");

    int num_tasks,
            taskID,
            num_workers,
            source,
            dest,
            rows,        /* rows of matrix A sent to each worker */
    ave_row,
            extra,
            offset,
            i,
            j,
            k,
            rc = 0;

    double a[NRA][NCA], /* matrix A to be multiplied */
    b[NCA][NCB],        /* matrix B to be multiplied */
    c[NRA][NCB];        /* result matrix C */

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskID);
    if (num_tasks < 2) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    num_workers = num_tasks - 1;
    if (taskID == MASTER) {
        printf("mpi_mm has started with %d tasks.\n", num_tasks);

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        srand(time(NULL));  // Set a seed for the random number generator
        for (i = 0; i < NRA; i++)
            for (j = 0; j < NCA; j++)
                a[i][j] = rand() % 100;
        for (i = 0; i < NCA; i++)
            for (j = 0; j < NCB; j++)
                b[i][j] = rand() % 100;
        print_matrix(&a[0][0], NRA, NCA, "Matrix A");
        print_matrix(&b[0][0], NCA, NCB, "Matrix B");

        ave_row = NRA / num_workers;
        extra = NRA % num_workers;
        offset = 0;
        for (dest = 1; dest <= num_workers; dest++) {
            rows = (dest <= extra) ? ave_row + 1 : ave_row;
            printf("Sending %d rows to task %d offset = %d\n", rows, dest, offset);
            MPI_Send(&offset, 1, MPI_INT, dest, FROM_MASTER,
                     MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, FROM_MASTER,
                     MPI_COMM_WORLD);
            MPI_Send(&a[offset][0], rows * NCA, MPI_DOUBLE, dest,
                     FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&b[0][0], NCA * NCB, MPI_DOUBLE, dest, FROM_MASTER,
                     MPI_COMM_WORLD);
            offset = offset + rows;
        }
        /* Receive results from worker tasks */
        for (source = 1; source <= num_workers; source++) {
            MPI_Recv(&offset, 1, MPI_INT, source, FROM_WORKER,
                     MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, FROM_WORKER,
                     MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset][0], rows * NCB, MPI_DOUBLE, source,
                     FROM_WORKER, MPI_COMM_WORLD, &status);
            printf("Received results from task %d\n", source); // id?
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        double executionTime = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        print_matrix(&c[0][0], NRA, NCB, "Result Matrix");
        printf("Done.\n");
        printf("Execution time: %f seconds\n", executionTime);
    }
        /******** worker task *****************/
    else { /* if (taskID > MASTER) */
        printf("mpi_mm task %d\n", taskID);

        MPI_Recv(&offset, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD,
                 &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&a[0][0], rows * NCA, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD,
                 &status);
        MPI_Recv(&b[0][0], NCA * NCB, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD,
                 &status);
        for (k = 0; k < NCB; k++)
            for (i = 0; i < rows; i++) {
                c[i][k] = 0.0;
                for (j = 0; j < NCA; j++)
                    c[i][k] = c[i][k] + a[i][j] * b[j][k];
            }
        MPI_Send(&offset, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&c[0][0], rows * NCB, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
    }
    MPI_Finalize();
}

void print_matrix(double *m, int nr, int nc, char *msg) {
    /* Print results */
    printf("****\n");
    printf("%s", msg);
    for (int i = 0; i < nr; i++) {
        printf("\n");
        for (int j = 0; j < nc; j++)
            printf("%6.2f ", m[i * nc + j]);
    }
    printf("\n********\n");
}

