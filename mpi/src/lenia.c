#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lenia.h"
#include "orbium.h"
#include "gifenc.h"


// Uncomment to generate gif animation
//#define GENERATE_GIF

// For prettier indexing syntax
#define w(r, c) (w[(r) * w_cols + (c)])
#define input(r, c) (input[(r) * cols + ((c) % cols)])

void exchange_overlap(double *padded_world, int n_rows, int cols, int rank, int procs)
{
    MPI_Request requests[4];
    MPI_Status statuses[4];
    int request_count = 0;

    // Send top overlap row to previous rank, receive from next rank
    int target_rank = rank - 1;
    target_rank = (target_rank + procs) % procs;
    MPI_Isend(padded_world + cols, cols, MPI_DOUBLE, target_rank, 1, MPI_COMM_WORLD, &requests[request_count++]);
    MPI_Irecv(padded_world, cols, MPI_DOUBLE, target_rank, 0, MPI_COMM_WORLD, &requests[request_count++]);

    // Send bottom overlap row to next rank, receive from previous rank
    target_rank = (rank + 1) % procs;
    MPI_Isend(padded_world + (n_rows - 2) * cols, cols, MPI_DOUBLE, target_rank, 0, MPI_COMM_WORLD, &requests[request_count++]);
    MPI_Irecv(padded_world + (n_rows - 1) * cols, cols, MPI_DOUBLE, target_rank, 1, MPI_COMM_WORLD, &requests[request_count++]);

    if (request_count > 0) {
        MPI_Waitall(request_count, requests, statuses);
    }
}

// Function to calculate Gaussian
inline double gauss(double x, double mu, double sigma)
{
    return exp(-0.5 * pow((x - mu) / sigma, 2));
}

// Function for growth criteria
double growth_lenia(double u)
{
    double mu = 0.15;
    double sigma = 0.015;
    return -1 + 2 * gauss(u, mu, sigma); // Baseline -1, peak +1
}

// Function to generate convolution kernel
double *generate_kernel(double *K, const unsigned int size)
{
    // Construct ring convolution filter
    double mu = 0.5;
    double sigma = 0.15;
    int r = size / 2;
    double sum = 0;
    if (K != NULL)
    {
        for (int y = -r; y < r; y++)
        {
            for (int x = -r; x < r; x++)
            {
                double distance = sqrt((1 + x) * (1 + x) + (1 + y) * (1 + y)) / r;
                K[(y + r) * size + x + r] = gauss(distance, mu, sigma);
                if (distance > 1)
                {
                    K[(y + r) * size + x + r] = 0; // Cut at d=1
                }
                sum += K[(y + r) * size + x + r];
            }
        }
        // Normalize
        for (unsigned int y = 0; y < size; y++)
        {
            for (unsigned int x = 0; x < size; x++)
            {
                K[y * size + x] /= sum;
            }
        }
    }
    return K;
}

// Function to perform convolution on input using kernel w
inline double *convolve2d(double *result, const double *input, const double *w, const unsigned int rows, const unsigned int cols, const unsigned int w_rows, const unsigned int w_cols)
{
    if (result != NULL && input != NULL && w != NULL)
    {
        for (unsigned int i = 0; i < rows; i++)
        {
            for (unsigned int j = 0; j < cols; j++)
            {
                double sum = 0;
                for (int ki = w_rows - 1, kri = 0; ki >= 0; ki--, kri++)
                {
                    for (int kj = w_cols - 1, kcj = 0; kj >= 0; kj--, kcj++)
                    {
                        sum += w(ki, kj) * input((i - w_rows / 2 + rows + kri), (j - w_cols / 2 + cols + kcj));
                    }
                }
                result[i * cols + j] = sum;
            }
        }
    }
    return result;
}

// Function to evolve Lenia
double *evolve_lenia(const unsigned int rows, const unsigned int cols, const unsigned int steps, const double dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums)
{
    int rank, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

#ifdef GENERATE_GIF
    ge_GIF *gif = ge_new_gif(
        "lenia.gif",     /* file name */
        cols, rows,      /* canvas size */
        inferno_pallete, /*pallete*/
        8,               /* palette depth == log2(# of colors) */
        -1,              /* no transparency */
        0                /* infinite loop */
    );
#endif

    // Allocate memory
    double *w = (double *)calloc(kernel_size * kernel_size, sizeof(double));

    // Generate convolution kernel
    w=generate_kernel(w,kernel_size);

    // Distribute rows to processes
    unsigned int n_rows = rows / procs;
    if (rank < rows % procs) {
        n_rows++;
    }

    double *padded_world = (double *)calloc((n_rows+2)* cols, sizeof(double));
    double *inner_world = padded_world + cols; // Skip top overlapping row
    double *tmp = (double *)calloc(n_rows * cols, sizeof(double));

    printf("Process %d handling rows %d to %d\n", rank, rank * (rows / procs) + (rank < rows % procs ? rank : rows % procs), rank * (rows / procs) + (rank < rows % procs ? rank : rows % procs) + n_rows - 1);

    // Place orbiums
    for (unsigned int o = 0; o < num_orbiums; o++)
    {
        int orbium_row = orbiums[o].row - rank * n_rows;
        // relevant but redundant condition
        // if (orbium_row >= -ORBIUM_SIZE && orbium_row < n_rows + ORBIUM_SIZE)
        padded_world = place_orbium(padded_world, n_rows, cols, orbium_row, orbiums[o].col, orbiums[o].angle);
    }
    
    // Lenia Simulation
    for (unsigned int step = 0; step < steps; step++)
    {
        // Exchange overlapping rows with neighbors
        exchange_overlap(padded_world, n_rows, cols, rank, procs);
        // Convolution
        tmp = convolve2d(tmp, inner_world, w, n_rows, cols, kernel_size, kernel_size);

        // Evolution
        for (unsigned int i = 0; i < n_rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                inner_world[i * cols + j] += dt * growth_lenia(tmp[i * cols + j]);
                inner_world[i * cols + j] = fmin(1, fmax(0, inner_world[i * cols + j])); // Clip between 0 and 1
#ifdef GENERATE_GIF
                gif->frame[i * cols + j] = inner_world[i * cols + j] * 255;
#endif
            }
        }
#ifdef GENERATE_GIF
        ge_add_frame(gif, 5);
#endif
    }
#ifdef GENERATE_GIF
    ge_close_gif(gif);
#endif
    free(w);
    free(tmp);
    // do not return inner_world because parent calls free()
    return padded_world;
}
