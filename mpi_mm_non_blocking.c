/*FILE: mpi_mm_non_blocking.c
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

    int numtasks,
            taskid,
            numworkers,
            source,
            dest,
            rows,        /* rows of matrix A sent to each worker */
    averow,
            extra,
            offset,
            i,
            j,
            k,
            rc = 0;

    double a[NRA][NCA], /* matrix A to be multiplied */
    b[NCA][NCB], /* matrix B to be multiplied */
    c[NRA][NCB]; /* result matrix C */

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    if (numtasks < 2) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    numworkers = numtasks - 1;
    if (taskid == MASTER) {
        printf("mpi_mm has started with %d tasks.\n", numtasks);

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

        averow = NRA / numworkers;
        extra = NRA % numworkers;
        offset = 0;
        MPI_Request send_req[3];
        for (dest = 1; dest <= numworkers; dest++) {
            rows = (dest <= extra) ? averow + 1 : averow;
            printf("Sending %d rows to task %d offset = %d\n", rows, dest, offset);
            MPI_Isend(&rows, 1, MPI_INT, dest, FROM_MASTER,
                      MPI_COMM_WORLD, &send_req[0]);
            MPI_Isend(&a[offset][0], rows * NCA, MPI_DOUBLE, dest,
                      FROM_MASTER, MPI_COMM_WORLD, &send_req[1]);
            MPI_Isend(&b[0][0], NCA * NCB, MPI_DOUBLE, dest, FROM_MASTER,
                      MPI_COMM_WORLD, &send_req[2]);
            MPI_Waitall(3, (MPI_Request *) &send_req, MPI_STATUS_IGNORE);
            printf("Sent %d rows to task %d offset = %d\n", rows, dest, offset);
            offset = offset + rows;
        }
        printf("All rows have been sent\n");

        /* Receive results from worker tasks */
        offset = 0;
        for (source = 1; source <= numworkers; source++) {
            rows = (source <= extra) ? averow + 1 : averow;
            MPI_Recv(&c[offset][0], rows * NCB, MPI_DOUBLE, source,
                     FROM_WORKER, MPI_COMM_WORLD, &status);
            offset = offset + rows;
            printf("Received %d rows from task %d offset = %d\n", rows, source, offset);
        }
        printf("All results has been received\n");

        clock_gettime(CLOCK_MONOTONIC, &end);
        double executionTime = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        print_matrix(&c[0][0], NRA, NCB, "Result Matrix");
        printf("Done.\n");
        printf("Execution time: %f seconds\n", executionTime);
    }
        /******** worker task *****************/
    else { /* if (taskid > MASTER) */
        printf("mpi_mm task %d\n", taskid);

        MPI_Request rec_req[2];
        MPI_Recv(&rows, 1, MPI_INT, MASTER, FROM_MASTER,
                 MPI_COMM_WORLD, &status);
        MPI_Irecv(&a[0][0], rows * NCA, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD,
                  &rec_req[0]);
        MPI_Irecv(&b[0][0], NCA * NCB, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD,
                  &rec_req[1]);

        MPI_Waitall(2, (MPI_Request *) &rec_req, MPI_STATUS_IGNORE);
        printf("Received %d rows by task %d\n", rows, taskid);

        for (k = 0; k < NCB; k++)
            for (i = 0; i < rows; i++) {
                c[i][k] = 0.0;
                for (j = 0; j < NCA; j++)
                    c[i][k] = c[i][k] + a[i][j] * b[j][k];
            }

        printf("Sending %d rows from task %d\n", rows, taskid);
        MPI_Send(&c[0][0], rows * NCB, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        printf("Sent %d rows from task %d\n", rows, taskid);
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


