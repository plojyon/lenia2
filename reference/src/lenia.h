#ifndef LENIA_H
#define LENIA_H

#include <mpi.h>

struct orbium_coo
{
    int row;
    int col;
    int angle;
};

double *evolve_lenia(unsigned int rows, unsigned int cols, const unsigned int steps, const double dt, const unsigned int kernel_size, struct orbium_coo *orbiums, const unsigned int num_orbiums);

#endif
