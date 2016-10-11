/**
 *
 * @file dsyrk.c
 *
 *  PLASMA computational routine.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley, Univ. of Colorado Denver and
 *  Univ. of Manchester.
 *
 * @version 3.0.0
 * @author Pedro V. Lara
 * @date 2016-05-24
 * @generated from compute/zsyrk.c, normal z -> d, Tue Jul 26 14:08:11 2016
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
 * @ingroup plasma_syrk
 *
 *  Performs one of the symmetric rank k operations
 *
 *    \f[ C = \alpha A \times A^T + \beta C \f],
 *    or
 *    \f[ C = \alpha A^T \times A + \beta C \f],
 *
 *  alpha and beta are real scalars, C is an n-by-n symmetric
 *  matrix and A is an n-by-k matrix in the first case and a k-by-n
 *  matrix in the second case.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of C is stored;
 *          - PlasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          - PlasmaNoTrans: \f[ C = \alpha A \times A^T + \beta C \f];
 *          - PlasmaTrans:   \f[ C = \alpha A^T \times A + \beta C \f].
 *
 * @param[in] n
 *          The order of the matrix C. n >= 0.
 *
 * @param[in] k
 *          The number of columns of the matrix op( A ).
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          A is a lda-by-ka matrix, where ka is k when trans = PlasmaNoTrans,
 *          and is n otherwise.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda must be at least
 *          max(1, n) if trans == PlasmaNoTrans, otherwise lda must
 *          be at least max(1, k).
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[in,out] C
 *          C is a ldc-by-n matrix.
 *          On exit, the array uplo part of the matrix is overwritten
 *          by the uplo part of the updated matrix.
 *
 * @param[in] ldc
 *          The leading dimension of the array C. ldc >= max(1, n).
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 *
 *******************************************************************************
 *
 * @sa PLASMA_dsyrk_Tile_Async
 * @sa PLASMA_csyrk
 * @sa PLASMA_dsyrk
 * @sa PLASMA_ssyrk
 *
 ******************************************************************************/
int PLASMA_dsyrk(PLASMA_enum uplo, PLASMA_enum trans, int n, int k,
                 double alpha, double *A, int lda,
                 double beta,  double *C, int ldc)
{
    int i;
    int Am, An;
    int nb;
    int retval;
    int status;

    PLASMA_desc descA;
    PLASMA_desc descC;

    // Get PLASMA context.
    //plasma_context_t *plasma = plasma_context_self();
    //if (plasma == NULL) {
    //    plasma_error("PLASMA not initialized");
    //    return PLASMA_ERR_NOT_INITIALIZED;
    //}

    // Check input arguments.
    if ((uplo != PlasmaUpper) &&
        (uplo != PlasmaLower)) {
        plasma_error("illegal value of uplo");
        return -1;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)) {
        plasma_error("illegal value of trans");
        return -2;
    }
    if (trans == PlasmaNoTrans) {
        Am = n; An = k;
    }
    else {
        Am = k; An = n;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -3;
    }
    if (k < 0) {
        plasma_error("illegal value of k");
        return -4;
    }
    if (lda < imax(1, Am)) {
        plasma_error("illegal value of lda");
        return -7;
    }
    if (ldc < imax(1, n)) {
        plasma_error("illegal value of ldc");
        return -10;
    }

    // quick return
    if (n == 0 ||
        ((alpha == (double)0.0 || k == 0.0) && beta == (double)1.0))
        return PLASMA_SUCCESS;

    // Tune
    // status = plasma_tune(PLASMA_FUNC_DSYRK, n, k, 0);
    // if (status != PLASMA_SUCCESS) {
    //     plasma_error("PLASMA_dsyrk", "plasma_tune() failed");
    //     return status;
    // }


    // Set NT & KT
    //nb = plasma->nb;
    //nb = TILE_SIZE;
    // Initialize tile matrix descriptors.

    clock_t tic,toc;
    double time;
    
    // Create sequence.
    PLASMA_sequence *sequence = NULL;
    retval = plasma_sequence_create(&sequence);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_sequence_create() failed");
        return retval;
    }

    // Initialize request.
    PLASMA_request request = PLASMA_REQUEST_INITIALIZER;

    printf("SIZE TILE TIME\n");
    for( nb = 2; nb < n-1; nb++){
        time = 0.0;
        for( i = 0; i < 10; i++){ 

            descA = plasma_desc_init(PlasmaRealDouble, nb, nb,
                             nb*nb, Am, An, 0, 0, Am, An);

            descC = plasma_desc_init(PlasmaRealDouble, nb, nb,
                             nb*nb, n, n, 0, 0, n, n);

            // Allocate matrices in tile layout.
            retval = plasma_desc_mat_alloc(&descA);
            if (retval != PLASMA_SUCCESS) {
                plasma_error("plasma_desc_mat_alloc() failed");
                return retval;
            }
    
            retval = plasma_desc_mat_alloc(&descC);
            if (retval != PLASMA_SUCCESS) {
                plasma_error("plasma_desc_mat_alloc() failed");
                plasma_desc_mat_free(&descA);
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
                if (sequence->status == PLASMA_SUCCESS)
                    PLASMA_dcm2ccrb_Async(C, ldc, &descC, sequence, &request);

                // Call the tile async function.
                if (sequence->status == PLASMA_SUCCESS) {
                    PLASMA_dsyrk_Tile_Async(uplo, trans,
                                            alpha, &descA,
                                            beta, &descC,
                                            sequence, &request);
                }

                // Translate back to LAPACK layout.
                if (sequence->status == PLASMA_SUCCESS)
                    PLASMA_dccrb2cm_Async(&descC, C, ldc, sequence, &request);
            } // pragma omp parallel block closed

            // Clock    
            toc = clock(); 
            time += (double)(toc - tic) / CLOCKS_PER_SEC;
    
            // Check for errors in the async execution
            if (sequence->status != PLASMA_SUCCESS)
                return sequence->status;

            // Free matrices in tile layout.
            plasma_desc_mat_free(&descA);
            plasma_desc_mat_free(&descC);

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
 * @ingroup plasma_syrk
 *
 *  Performs rank k update.
 *  Non-blocking tile version of PLASMA_dsyrk().
 *  May return before the computation is finished.
 *  Operates on matrices stored by tiles.
 *  Tile equivalent of PLASMA_dsyrk().
 *  Operates on matrices stored by tiles.
 *  All matrices are passed through descriptors.
 *  All dimensions are taken from the descriptors.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of C is stored;
 *          - PlasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          - PlasmaNoTrans: \f[ C = \alpha A \times A^T + \beta C \f];
 *          - PlasmaTrans:   \f[ C = \alpha A^T \times A + \beta C \f].
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          Descriptor of matrix A.
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] C
 *          Descriptor of matrix C.
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
 * @sa PLASMA_dsyrk
 * @sa PLASMA_dsyrk_Tile_Async
 * @sa PLASMA_csyrk_Tile_Async
 * @sa PLASMA_dsyrk_Tile_Async
 * @sa PLASMA_ssyrk_Tile_Async
 *
 ******************************************************************************/
void PLASMA_dsyrk_Tile_Async(PLASMA_enum uplo, PLASMA_enum trans,
                            double alpha, PLASMA_desc *A,
                            double beta,  PLASMA_desc *C,
                            PLASMA_sequence *sequence, PLASMA_request *request)
{
    // Get PLASMA context.
    //plasma_context_t *plasma = plasma_context_self();
    //if (plasma == NULL) {
    //    plasma_error("PLASMA not initialized");
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
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)) {
        plasma_error("illegal value of trans");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        plasma_error("invalid A");
        return;
    }
    if (plasma_desc_check(C) != PLASMA_SUCCESS) {
        plasma_error("invalid C");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (sequence == NULL) {
        plasma_error("NULL sequence");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (request == NULL) {
        plasma_error("NULL request");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    int Am, An, Amb;

    if (trans == PlasmaNoTrans) {
        Am  = A->m;
        An  = A->n;
        Amb = A->mb;
    }
    else {
        Am  = A->n;
        An  = A->m;
        Amb = A->nb;
    }

    if (C->mb != C->nb) {
        plasma_error("only square tiles supported");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (Amb != C->mb) {
        plasma_error("tile sizes have to match");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (C->m != C->n) {
        plasma_error("only square matrix C is supported");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (Am != C->m) {
        plasma_error("size of matrices have to match");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check sequence status.
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    // quick return
    if (C->m == 0 ||
        ((alpha == 0.0 || An == 0) && beta == 1.0))
        return;

    // Call the parallel function.
    plasma_pdsyrk(uplo, trans,
                  alpha, *A,
                  beta, *C,
                  sequence, request);

    return;
}
