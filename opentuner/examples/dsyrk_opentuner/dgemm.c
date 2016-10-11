/**
 *
 * @file dgemm.c
 *
 *  PLASMA computational routine.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 * @generated from compute/zgemm.c, normal z -> d, Mon Jul 11 15:47:17 2016
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
 * @ingroup plasma_gemm
 *
 *  Performs one of the matrix-matrix operations
 *
 *          \f[ C = \alpha [op( A )\times op( B )] + \beta C, \f]
 *
 *  where op( X ) is one of:
 *          - op( X ) = X   or
 *          - op( X ) = X^T or
 *          - op( X ) = X^H
 *
 *  alpha and beta are scalars, and A, B and C are matrices, with op( A )
 *  an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
 *
 *******************************************************************************
 *
 * @param[in] transA
 *          - PlasmaNoTrans:   A is not transposed,
 *          - PlasmaTrans:     A is transposed,
 *          - PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] transB
 *          - PlasmaNoTrans:   B is not transposed,
 *          - PlasmaTrans:     B is transposed,
 *          - PlasmaConjTrans: B is conjugate transposed.
 *
 * @param[in] m
 *          The number of rows of the matrix op( A ) and of the matrix C.
 *          m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix op( B ) and of the matrix C.
 *          n >= 0.
 *
 * @param[in] k
 *          The number of columns of the matrix op( A ) and the number of rows
 *          of the matrix op( B ). k >= 0.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          An lda-by-ka matrix, where ka is k when transA = PlasmaNoTrans,
 *          and is m otherwise.
 *
 * @param[in] lda
 *          The leading dimension of the array A.
 *          When transA = PlasmaNoTrans, lda >= max(1,m),
 *          otherwise, lda >= max(1,k).
 *
 * @param[in] B
 *          An ldb-by-kb matrix, where kb is n when transB = PlasmaNoTrans,
 *          and is k otherwise.
 *
 * @param[in] ldb
 *          The leading dimension of the array B.
 *          When transB = PlasmaNoTrans, ldb >= max(1,k),
 *          otherwise, ldb >= max(1,n).
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] C
 *          An ldc-by-n matrix. On exit, the array is overwritten by the m-by-n
 *          matrix ( alpha*op( A )*op( B ) + beta*C ).
 *
 * @param[in] ldc
 *          The leading dimension of the array C. ldc >= max(1,m).
 *
 *******************************************************************************
 *
 * @retval PLASMA_SUCCESS successful exit
 *
 *******************************************************************************
 *
 * @sa PLASMA_dgemm_Tile_Async
 * @sa PLASMA_cgemm
 * @sa PLASMA_dgemm
 * @sa PLASMA_sgemm
 *
 ******************************************************************************/
int PLASMA_dgemm(PLASMA_enum transA, PLASMA_enum transB,
                 int m, int n, int k,
                 double alpha, double *A, int lda,
                                           double *B, int ldb,
                 double beta,  double *C, int ldc)
{

    int Am, An;
    int Bm, Bn;
    int nb;
    int retval;
    int status;

    PLASMA_desc descA;
    PLASMA_desc descB;
    PLASMA_desc descC;

    double zzero = 0.0;
    double zone  = 1.0;

    // Get PLASMA context.
    //plasma_context_t *plasma = plasma_context_self();
    //if (plasma == NULL) {
    //    plasma_error("PLASMA not initialized");
    //    return PLASMA_ERR_NOT_INITIALIZED;
    //}

    // Check input arguments.
    if ((transA != PlasmaNoTrans) &&
        (transA != PlasmaTrans) &&
        (transA != PlasmaConjTrans)) {
        plasma_error("illegal value of transA");
        return -1;
    }
    if ((transB != PlasmaNoTrans) &&
        (transB != PlasmaTrans) &&
        (transB != PlasmaConjTrans)) {
        plasma_error("illegal value of transB");
        return -2;
    }
    if (m < 0) {
        plasma_error("illegal value of m");
        return -3;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -4;
    }
    if (k < 0) {
        plasma_error("illegal value of k");
        return -5;
    }
    if (A == NULL) {
        plasma_error("NULL A");
        return -7;
    }

    if (transA == PlasmaNoTrans) {
        Am = m;
        An = k;
    }
    else {
        Am = k;
        An = m;
    }
    if (transB == PlasmaNoTrans) {
        Bm = k;
        Bn = n;
    }
    else {
        Bm = n;
        Bn = k;
    }

    if (lda < imax(1, Am)) {
        plasma_error("illegal value of lda");
        return -8;
    }
    if (B == NULL) {
        plasma_error("NULL B");
        return -9;
    }
    if (ldb < imax(1, Bm)) {
        plasma_error("illegal value of ldb");
        return -10;
    }
    if (C == NULL) {
        plasma_error("NULL C");
        return -12;
    }
    if (ldc < imax(1, m)) {
        plasma_error("illegal value of ldc");
        return -13;
    }

    // quick return
    if (m == 0 || n == 0 || ((alpha == zzero || k == 0) && beta == zone))
        return PLASMA_SUCCESS;

    // Tune.
    // if (plasma_tune(PLASMA_FUNC_DGEMM, m, n, 0) != PLASMA_SUCCESS) {
    //     plasma_error("plasma_tune() failed");
    //     return status;
    // }
    //nb = plasma->nb;
    nb = TILE_SIZE;
    //printf("TILE SIZE = %d\n", TILE_SIZE);

    // Initialize tile matrix descriptors.
    descA = plasma_desc_init(PlasmaRealDouble, nb, nb,
                             nb*nb, Am, An, 0, 0, Am, An);

    descB = plasma_desc_init(PlasmaRealDouble, nb, nb,
                             nb*nb, Bm, Bn, 0, 0, Bm, Bn);

    descC = plasma_desc_init(PlasmaRealDouble, nb, nb,
                             nb*nb, m, n, 0, 0, m, n);

    // Allocate matrices in tile layout.
    retval = plasma_desc_mat_alloc(&descA);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        return retval;
    }
    retval = plasma_desc_mat_alloc(&descB);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        plasma_desc_mat_free(&descA);
        return retval;
    }
    retval = plasma_desc_mat_alloc(&descC);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_desc_mat_alloc() failed");
        plasma_desc_mat_free(&descA);
        plasma_desc_mat_free(&descB);
        return retval;
    }

    // Create sequence.
    PLASMA_sequence *sequence = NULL;
    retval = plasma_sequence_create(&sequence);
    if (retval != PLASMA_SUCCESS) {
        plasma_error("plasma_sequence_create() failed");
        return retval;
    }
    // Initialize request.
    PLASMA_request request = PLASMA_REQUEST_INITIALIZER;

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
            PLASMA_dcm2ccrb_Async(B, ldb, &descB, sequence, &request);
        if (sequence->status == PLASMA_SUCCESS)
            PLASMA_dcm2ccrb_Async(C, ldc, &descC, sequence, &request);

        // Call the tile async function.
        if (sequence->status == PLASMA_SUCCESS) {
            PLASMA_dgemm_Tile_Async(transA, transB,
                                    alpha, &descA,
                                    &descB,
                                    beta, &descC,
                                    sequence, &request);
        }

        // Translate back to LAPACK layout.
        if (sequence->status == PLASMA_SUCCESS)
            PLASMA_dccrb2cm_Async(&descC, C, ldc, sequence, &request);
    } // pragma omp parallel block closed

    // Check for errors in the async execution
    if (sequence->status != PLASMA_SUCCESS)
        return sequence->status;

    // Free matrices in tile layout.
    plasma_desc_mat_free(&descA);
    plasma_desc_mat_free(&descB);
    plasma_desc_mat_free(&descC);

    // Return status.
    status = sequence->status;
    plasma_sequence_destroy(sequence);
    return status;

}

/***************************************************************************//**
 *
 * @ingroup plasma_gemm
 *
 *  Performs matrix multiplication.
 *  Non-blocking tile version of PLASMA_dgemm().
 *  May return before the computation is finished.
 *  Operates on matrices stored by tiles.
 *  All matrices are passed through descriptors.
 *  All dimensions are taken from the descriptors.
 *  Allows for pipelining of operations at runtime.
 *
 *******************************************************************************
 *
 * @param[in] transA
 *          - PlasmaNoTrans:   A is not transposed,
 *          - PlasmaTrans:     A is transposed,
 *          - PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] transB
 *          - PlasmaNoTrans:   B is not transposed,
 *          - PlasmaTrans:     B is transposed,
 *          - PlasmaConjTrans: B is conjugate transposed.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          Descriptor of matrix A.
 *
 * @param[in] B
 *          Descriptor of matrix B.
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
 * @sa PLASMA_dgemm
 * @sa PLASMA_cgemm_Tile_Async
 * @sa PLASMA_dgemm_Tile_Async
 * @sa PLASMA_sgemm_Tile_Async
 *
 ******************************************************************************/
void PLASMA_dgemm_Tile_Async(PLASMA_enum transA, PLASMA_enum transB,
                            double alpha, PLASMA_desc *A,
                                                      PLASMA_desc *B,
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
    if ((transA != PlasmaNoTrans) &&
        (transA != PlasmaTrans) &&
        (transA != PlasmaConjTrans)) {
        plasma_error("illegal value of transA");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if ((transB != PlasmaNoTrans) &&
        (transB != PlasmaTrans) &&
        (transB != PlasmaConjTrans)) {
        plasma_error("illegal value of transB");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        plasma_error("invalid A");
        return;
    }
    if (plasma_desc_check(B) != PLASMA_SUCCESS) {
        plasma_error("invalid B");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
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

    int Am, An, Ai, Aj, Amb, Anb;
    int Bm, Bn, Bi, Bj, Bmb, Bnb;

    if (transA == PlasmaNoTrans) {
        Am  = A->m;
        An  = A->n;
        Amb = A->mb;
        Anb = A->nb;
        Ai  = A->i;
        Aj  = A->j;
    }
    else {
        Am  = A->n;
        An  = A->m;
        Amb = A->nb;
        Anb = A->mb;
        Ai  = A->j;
        Aj  = A->i;
    }
    if (transB == PlasmaNoTrans) {
        Bm  = B->m;
        Bn  = B->n;
        Bmb = B->mb;
        Bnb = B->nb;
        Bi  = B->i;
        Bj  = B->j;
    }
    else {
        Bm  = B->n;
        Bn  = B->m;
        Bmb = B->nb;
        Bnb = B->mb;
        Bi  = B->j;
        Bj  = B->i;
    }

    if (Amb != C->mb || Anb != Bmb || Bnb != C->nb) {
        plasma_error("tile size mismatch");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (Am != C->m || An != Bm || Bn != C->n) {
        plasma_error("matrix size mismatch");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (Ai%Amb != C->i%C->mb ||
        Bj%Bnb != C->j%C->nb || Aj%Anb != Bi%Bmb) {
        plasma_error("start indexes have to match");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }

    // Check sequence status.
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    // quick return
    if (C->m == 0 || C->n == 0 || ((alpha == 0.0 || An == 0) && beta == 1.0))
        return;

    // Call the parallel function.
    plasma_pdgemm(transA, transB,
                  alpha, *A,
                         *B,
                   beta, *C,
                  sequence, request);

    return;
}
