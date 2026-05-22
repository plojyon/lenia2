#include <stdio.h>
#include <stdlib.h>
#include "lenia.h"

#define DT 0.1
#define NUM_STEPS 100
#define NUM_ORBIUMS 2
#define KERNEL_SIZE 26

int main(int argc, char *argv[])
{
    int myid, procs;
    char node_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    if (argc < 2)
    {
        printf("usage: <N>");
    }

    int n = atoi(argv[1]);

    printf("N: %d", n);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);         // process ID
    MPI_Comm_size(MPI_COMM_WORLD, &procs);        // number of processes involved in communication
    MPI_Get_processor_name(node_name, &name_len); // compute node name
    printf("Hello from process %d of %d in node %s\n", myid, procs, node_name);

    struct orbium_coo orbiums[NUM_ORBIUMS] = {{0, n / 3, 0}, {n / 3, 0, 180}};

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    double *world = evolve_lenia(n, n, NUM_STEPS, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS);

    double local_time = MPI_Wtime() - start;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (myid == 0)
    {
        printf("Total execution time: %.3f\n", max_time);
    }
    free(world);
    MPI_Finalize();
    return 0;
}