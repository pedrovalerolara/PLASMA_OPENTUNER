#include "test.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "plasma.h"

#define M 512
#define N 512
#define K 512

//Flashing cache
#define M_f 512
#define N_f 512
#define K_f 512


int main(int argc, char **argv)
{
    int i,n,k;
        
    n = N;
    k = K;
    
    double complex alpha = 2.0 + 1.0*I;
    double complex beta  = 2.0 + 1.0*I;
    
    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    double *A =
        (double*)malloc((size_t)n*k*sizeof(double));
    assert(A != NULL);

    double *C =
        (double*)malloc((size_t)n*n*sizeof(double));
    assert(C != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_dlarnv(1, seed, (size_t)n*k, A);
    assert(retval == 0);

    retval = LAPACKE_dlarnv(1, seed, (size_t)n*n, C);
    assert(retval == 0);

    //================================================================
    // Flashing caches
    //================================================================
   
    int m_f,n_f,k_f;
        
    m_f = M_f;
    n_f = N_f;
    k_f = K_f;
    
    double *A_f =
        (double*)malloc((size_t)m_f*k_f*sizeof(double));
    assert(A_f != NULL);

    double *B_f =
        (double*)malloc((size_t)k_f*n_f*sizeof(double));
    assert(B_f != NULL);

    double *C_f =
        (double*)malloc((size_t)m_f*n_f*sizeof(double));
    assert(C_f != NULL);

    //int seed[] = {0, 0, 0, 1};
    //lapack_int retval;
    retval = LAPACKE_dlarnv(1, seed, (size_t)m_f*k_f, A_f);
    assert(retval == 0);

    retval = LAPACKE_dlarnv(1, seed, (size_t)k_f*n_f, B_f);
    assert(retval == 0);

    retval = LAPACKE_dlarnv(1, seed, (size_t)m_f*n_f, C_f);
    assert(retval == 0);
    
    PLASMA_dgemm(
            PlasmaNoTrans, PlasmaNoTrans,
            m_f, n_f, k_f,
            alpha, A_f, m_f,
                   B_f, k_f,
            beta, C_f, m_f);

    //================================================================
    // Run and time PLASMA.
    //================================================================
    
    for(i=0; i<10; i++){    

        PLASMA_dsyrk(
            PlasmaLower, PlasmaNoTrans,
            n, k,
            alpha, A, n,
            beta, C, n);
 
    }

    free(A);
    free(C);
    free(A_f);
    free(B_f);
    free(C_f);
    
}


 
