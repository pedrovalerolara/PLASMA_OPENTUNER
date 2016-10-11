/**
 *
 * @file dpotrf.c
 *
 *  PLASMA computational routines
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley, Univ. of Colorado Denver and
 *  Univ. of Manchester.
 *
 * @version 3.0.0
 * @author Pedro V. Lara
 * @date
 * @generated from compute/zpotrf.c, normal z -> d, Tue Jul 26 14:08:10 2016
 *
 **/

#include "plasma_types.h"
#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_d.h"

/***************************************************************************//**
 *
 * @ingroup plasma_potrf
 *
 *  Performs the Cholesky factorization of a symmetric positive definite
 *  (or symmetric positive definite in the complex case) matrix A.
 *  The factorization has the form
 *
 *    \f[ A = L \times L^H \f],
 *    or
 *    \f[ A = U^H \times U \f],
 *
 *  where U is an upper triangular matrix and L is a lower triangular matrix.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          On entry, the symmetric positive definite (or symmetric) matrix A.
 *          If uplo = PlasmaUpper, the leading N-by-N upper triangular part of A
 *          contains the upper triangular part of the matrix A, and the strictly lower triangular
 *          part of A is not referenced.
 *          If uplo = 'L', the leading N-by-N lower triangular part of A contains the lower
 *          triangular part of the matrix A, and the strictly upper triangular part of A is not
 *          referenced.
 *          On exit, if return value = 0, the factor U or L from the Cholesky factorization
 *          A = U^H*U or A = L*L^H.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,n).
 *
 * *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 *
 *******************************************************************************
 *
 * @sa PLASMA_dpotrf_Tile_Async
 * @sa PLASMA_cpotrf
 * @sa PLASMA_dpotrf
 * @sa PLASMA_spotrf
 *
 ******************************************************************************/
int PLASMA_dpotrf(PLASMA_enum uplo, int n,
                  double *A, int lda)
{
    int i;
    int nb;
    int retval;
    int status;

    PLASMA_desc descA;

    // Get PLASMA context.
    //plasma_context_t *plasma = plasma_context_self();
    //if (plasma == NULL) {
    //    plasma_fatal_error("PLASMA not initialized");
    //    return PLASMA_ERR_NOT_INITIALIZED;
    //}

    // Check input arguments
    if ((uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        return -1;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -2;
    }
    if (lda < imax(1, n)) {
        plasma_error("illegal value of lda");
        return -4;
    }

    // quick return
    if (imax(n, 0) == 0)
        return PLASMA_SUCCESS;

    // Tune
    // status = plasma_tune(PLASMA_FUNC_DPOSV, N, N, 0);
    // if (status != PLASMA_SUCCESS) {
    //     plasma_error("plasma_tune() failed");
    //     return status;
    // }

    // Create sequence.
    PLASMA_sequence *sequence = NULL;
    retval = plasma_sequence_create(&sequence);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_sequence_create() failed");
        return retval;
    }
    // Initialize request.
    PLASMA_request request = PLASMA_REQUEST_INITIALIZER;

    // Set NT & KT
    //nb = plasma->nb;
    //nb = TILE_SIZE;

    clock_t tic,toc;
    double time;

    printf("SIZE TILE TIME\n");
    for( nb = 2; nb < n-1; nb++){
        time = 0.0;
        for( i = 0; i < 10; i++){ 

            // Initialize tile matrix descriptors.
            descA = plasma_desc_init(PlasmaRealDouble, nb, nb,
                                     nb*nb, n, n, 0, 0, n, n);

            // Allocate matrices in tile layout.
            retval = plasma_desc_mat_alloc(&descA);
            if (retval != PLASMA_SUCCESS) {
                plasma_error("plasma_desc_mat_alloc() failed");
                return retval;
            }

    
            // Clock
            tic = clock();

#pragma omp parallel
#pragma omp master
            {
                // the Async functions are submitted here.  If an error occurs
                // (at submission time or at run time) the sequence->status
                // will be marked with an error.  After an error, the next
                // Async will not _insert_ more tasks into the runtime.  The
                // sequence->status can be checked after each call to _Async
                // or at the end of the parallel region.

                // Translate to tile layout.
                PLASMA_dcm2ccrb_Async(A, lda, &descA, sequence, &request);

                // Call the tile async function.
                if (sequence->status == PLASMA_SUCCESS) {
                    PLASMA_dpotrf_Tile_Async(uplo, &descA, sequence, &request);
                }

                // Translate back to LAPACK layout.
                if (sequence->status == PLASMA_SUCCESS)
                    PLASMA_dccrb2cm_Async(&descA, A, lda, sequence, &request);
            } // pragma omp parallel block closed

            // Clock    
            toc = clock(); 
            time += (double)(toc - tic) / CLOCKS_PER_SEC;

            // Check for errors in the async execution
            if (sequence->status != PLASMA_SUCCESS)
                return sequence->status;

            // Free matrix A in tile layout.
            plasma_desc_mat_free(&descA);

        }// For i
       
        printf(" %d  %d  %0.4f\n", n, nb, time/10.);

    }// For nb

    // Return status.
    status = sequence->status;
    plasma_sequence_destroy(sequence);
    return status;
}

/***************************************************************************//**
 *
 * @ingroup plasma_potrf
 *
 *  Performs the Cholesky factorization of a symmetric positive definite
 *  or symmetric positive definite matrix.
 *  Non-blocking tile version of PLASMA_dpotrf().
 *  May return before the computation is finished.
 *  Operates on matrices stored by tiles.
 *  All matrices are passed through descriptors.
 *  All dimensions are taken from the descriptors.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] A
 *          On entry, the symmetric positive definite (or symmetric) matrix A.
 *          If uplo = PlasmaUpper, the leading n-by-n upper triangular part of A
 *          contains the upper triangular part of the matrix A, and the strictly lower triangular
 *          part of A is not referenced.
 *          If uplo = 'L', the leading n-by-n lower triangular part of A contains the lower
 *          triangular part of the matrix A, and the strictly upper triangular part of A is not
 *          referenced.
 *          On exit, if return value = 0, the factor U or L from the Cholesky factorization
 *          A = U^H*U or A = L*L^H.
 *
 * @param[in] sequence
 *          Identifies the sequence of function calls that this call belongs to
 *          (for completion checks and exception handling purposes).  Check
 *          the sequence->status for errors.
 *
 * @param[out] request
 *          Identifies this function call (for exception handling purposes).
 *
 * @retval void
 *          Errors are returned by setting sequence->status and
 *          request->status to error values.  The sequence->status and
 *          request->status should never be set to PLASMA_SUCCESS (the
 *          initial values) since another async call may be setting a
 *          failure value at the same time.
 *
 *******************************************************************************
 *
 * @sa PLASMA_dpotrf
 * @sa PLASMA_dpotrf_Tile_Async
 * @sa PLASMA_cpotrf_Tile_Async
 * @sa PLASMA_dpotrf_Tile_Async
 * @sa PLASMA_spotrf_Tile_Async
 *
 ******************************************************************************/
void PLASMA_dpotrf_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A,
                              PLASMA_sequence *sequence, PLASMA_request *request)
{
    // Get PLASMA context.
    //plasma_context_t *plasma = plasma_context_self();
    //if (plasma == NULL) {
    //    plasma_fatal_error("PLASMA not initialized");
    //    plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
    //    return;
    //}

    // Check input arguments.
    if ((uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        plasma_error("invalid A");
        return;
    }
    if (sequence == NULL) {
        plasma_fatal_error("NULL sequence");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (request == NULL) {
        plasma_fatal_error("NULL request");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    if (A->mb != A->nb) {
        plasma_error("only square tiles supported");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (A->m != A->n) {
        plasma_error("only square matrix A is supported");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check sequence status.
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    // quick return
    if (A->m == 0)
        return;

    // Call the parallel function.
    plasma_pdpotrf(uplo, *A, sequence, request);

    return;
}
