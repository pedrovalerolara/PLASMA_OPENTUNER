/**
 *
 * @file pdgemm.c
 *
 *  PLASMA computational routine.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @author Mark Gates
 * @date 2016-01-01
 * @generated from compute/pzgemm.c, normal z -> d, Mon Jul 11 15:47:18 2016
 *
 **/

#include "plasma_async.h"
#include "plasma_descriptor.h"
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_blas_d.h"

#define A(m, n) ((double*) plasma_getaddr(A, m, n))
#define B(m, n) ((double*) plasma_getaddr(B, m, n))
#define C(m, n) ((double*) plasma_getaddr(C, m, n))
/***************************************************************************//**
 * Parallel tile matrix-matrix multiplication.
 * @see PLASMA_dgemm_Tile_Async
 ******************************************************************************/
void plasma_pdgemm(PLASMA_enum transA, PLASMA_enum transB,
                   double alpha, PLASMA_desc A,
                                             PLASMA_desc B,
                   double beta,  PLASMA_desc C,
                   PLASMA_sequence *sequence, PLASMA_request *request)
{
    int m, n, k;
    int ldam, ldak, ldbn, ldbk, ldcm;
    int tempmm, tempnn, tempkn, tempkm;

    double zbeta;
    double zone = 1.0;

    if (sequence->status != PLASMA_SUCCESS)
        return;

    int innerK = (transA == PlasmaNoTrans ? A.n : A.m);

    for (m = 0; m < C.mt; m++) {
        tempmm = m == C.mt-1 ? C.m-m*C.mb : C.mb;
        ldcm = BLKLDD(C, m);
        for (n = 0; n < C.nt; n++) {
            tempnn = n == C.nt-1 ? C.n-n*C.nb : C.nb;
            if (alpha == 0.0 || innerK == 0) {
                //=======================================
                // alpha*A*B does not contribute; scale C
                //=======================================
                ldam = imax( 1, BLKLDD(A, 0) );
                ldbk = imax( 1, BLKLDD(B, 0) );
                CORE_OMP_dgemm(
                    transA, transB,
                    tempmm, tempnn, 0,
                    alpha, A(0, 0), ldam,
                           B(0, 0), ldbk,
                    beta,  C(m, n), ldcm);
            }
            else if (transA == PlasmaNoTrans) {
                ldam = BLKLDD(A, m);
                //=======================================
                // A: PlasmaNoTrans / B: PlasmaNoTrans
                //=======================================
                if (transB == PlasmaNoTrans) {
                    for (k = 0; k < A.nt; k++) {
                        tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                        ldbk = BLKLDD(B, k);
                        zbeta = k == 0 ? beta : zone;
                        CORE_OMP_dgemm(
                            transA, transB,
                            tempmm, tempnn, tempkn,
                            alpha, A(m, k), ldam,
                                   B(k, n), ldbk,
                            zbeta, C(m, n), ldcm);
                    }
                }
                //==========================================
                // A: PlasmaNoTrans / B: Plasma[Conj]Trans
                //==========================================
                else {
                    ldbn = BLKLDD(B, n);
                    for (k = 0; k < A.nt; k++) {
                        tempkn = k == A.nt-1 ? A.n-k*A.nb : A.nb;
                        zbeta = k == 0 ? beta : zone;
                        CORE_OMP_dgemm(
                            transA, transB,
                            tempmm, tempnn, tempkn,
                            alpha, A(m, k), ldam,
                                   B(n, k), ldbn,
                            zbeta, C(m, n), ldcm);
                    }
                }
            }
            else {
                //==========================================
                // A: Plasma[Conj]Trans / B: PlasmaNoTrans
                //==========================================
                if (transB == PlasmaNoTrans) {
                    for (k = 0; k < A.mt; k++) {
                        tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
                        ldak = BLKLDD(A, k);
                        ldbk = BLKLDD(B, k);
                        zbeta = k == 0 ? beta : zone;
                        CORE_OMP_dgemm(
                            transA, transB,
                            tempmm, tempnn, tempkm,
                            alpha, A(k, m), ldak,
                                   B(k, n), ldbk,
                            zbeta, C(m, n), ldcm);
                    }
                }
                //==============================================
                // A: Plasma[Conj]Trans / B: Plasma[Conj]Trans
                //==============================================
                else {
                    ldbn = BLKLDD(B, n);
                    for (k = 0; k < A.mt; k++) {
                        tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
                        ldak = BLKLDD(A, k);
                        zbeta = k == 0 ? beta : zone;
                        CORE_OMP_dgemm(
                            transA, transB,
                            tempmm, tempnn, tempkm,
                            alpha, A(k, m), ldak,
                                   B(n, k), ldbn,
                            zbeta, C(m, n), ldcm);
                    }
                }
            }
        }
    }
}
