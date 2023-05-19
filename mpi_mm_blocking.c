/*FILE: mpi_mm_blocking.c
* by Blaise Barney
* modified by Oleksandr Popov
*/

/* Run with mpirun --use-hwthread-cpus -n 4 lab4 */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define NRA 62          /* number of rows in matrix A */
#define NCA 15          /* number of columns in matrix A */
#define NCB 7           /* number of columns in matrix B */
#define MASTER 0        /* taskID of first task */
#define FROM_MASTER 1   /* setting a message type */
#define FROM_WORKER 2   /* setting a message type */

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
        for (i = 0; i < NRA; i++)
            for (j = 0; j < NCA; j++)
                a[i][j] = 10;
        for (i = 0; i < NCA; i++)
            for (j = 0; j < NCB; j++)
                b[i][j] = 10;

        ave_row = NRA / num_workers;
        extra = NRA % num_workers;
        offset = 0;
        for (dest = 1; dest <= num_workers; dest++) {
            rows = (dest <= extra) ? ave_row + 1 : ave_row;
            printf("Sending %d rows to task %d offset= % d\n", rows, dest, offset);
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
            printf("Received results from task %d\n", taskID); // id?
        }

        /* Print results */
        printf("****\n");
        printf("Result Matrix:\n");
        for (i = 0; i < NRA; i++) {
            printf("\n");
            for (j = 0; j < NCB; j++)
                printf("%6.2f ", c[i][j]);
        }
        printf("\n********\n");
        printf("Done.\n");
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

