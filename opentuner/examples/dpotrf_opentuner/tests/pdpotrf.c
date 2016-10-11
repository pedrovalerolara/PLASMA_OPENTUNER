/**
 *
 * @file pdpotrf.c
 *
 *  PLASMA computational routine.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley, Univ. of Colorado Denver and
 *  Univ. of Manchester.
 *
 * @version
 * @author Pedro V. Lara
 * @date
 * @generated from compute/pzpotrf.c, normal z -> d, Tue Jul 26 14:08:10 2016
 *
 **/

#include "plasma_async.h"
#include "plasma_descriptor.h"
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_blas_d.h"

#define A(m, n) ((double*) plasma_getaddr(A, m, n))
/***************************************************************************//**
 *  Parallel tile Cholesky factorization.
 * @see PLASMA_dpotrf_Tile_Async
 ******************************************************************************/
void plasma_pdpotrf(PLASMA_enum uplo, PLASMA_desc A,
                    PLASMA_sequence *sequence, PLASMA_request *request)
{
    int k, m, n;
    int ldak, ldam, ldan;
    int tempkm, tempmm;

    double zone  = (double) 1.0;
    double mzone = (double)-1.0;

    if (sequence->status != PLASMA_SUCCESS)
        return;

    //=======================================
    // PlasmaLower
    //=======================================
    if (uplo == PlasmaLower) {
        for (k = 0; k < A.mt; k++) {
            tempkm = k == A.mt-1 ? A.m-k*A.mb : A.mb;
            ldak = BLKLDD(A, k);
            CORE_OMP_dpotrf(
                PlasmaLower, tempkm,
                A(k, k), ldak);
            for (m = k+1; m < A.mt; m++) {
                tempmm = m == A.mt-1 ? A.m-m*A.mb : A.mb;
                ldam = BLKLDD(A, m);
                CORE_OMP_dtrsm(
                    PlasmaRight, PlasmaLower,
                    PlasmaConjTrans, PlasmaNonUnit,
                    tempmm, A.mb,
                    zone, A(k, k), ldak,
                          A(m, k), ldam);
            }
            for (m = k+1; m < A.mt; m++) {
                tempmm = m == A.mt-1 ? A.m-m*A.mb : A.mb;
                ldam = BLKLDD(A, m);
                CORE_OMP_dsyrk(
                    PlasmaLower, PlasmaNoTrans,
                    tempmm, A.mb,
                    -1.0, A(m, k), ldam,
                     1.0, A(m, m), ldam);
                for (n = k+1; n < m; n++) {
                    ldan = BLKLDD(A, n);
                    CORE_OMP_dgemm(
                        PlasmaNoTrans, PlasmaConjTrans,
                        tempmm, A.mb, A.mb,
                        mzone, A(m, k), ldam,
                               A(n, k), ldan,
                        zone,  A(m, n), ldam);
                }
            }
        }
    }
    //=======================================
    // PlasmaUpper
    //=======================================
    else {
        for (k = 0; k < A.nt; k++) {
            tempkm = k == A.nt-1 ? A.n-k*A.nb : A.nb;
            ldak = BLKLDD(A, k);
            CORE_OMP_dpotrf(
                PlasmaUpper, tempkm,
                A(k, k), ldak);
            for (m = k+1; m < A.nt; m++) {
                tempmm = m == A.nt-1 ? A.n-m*A.nb : A.nb;
                CORE_OMP_dtrsm(
                    PlasmaLeft, PlasmaUpper,
                    PlasmaConjTrans, PlasmaNonUnit,
                    A.nb, tempmm,
                    zone, A(k, k), ldak,
                          A(k, m), ldak);
            }
            for (m = k+1; m < A.nt; m++) {
                tempmm = m == A.nt-1 ? A.n-m*A.nb : A.nb;
                ldam = BLKLDD(A, m);
                CORE_OMP_dsyrk(
                    PlasmaUpper, PlasmaConjTrans,
                    tempmm, A.mb,
                    -1.0, A(k, m), ldak,
                     1.0, A(m, m), ldam);
                for (n = k+1; n < m; n++) {
                    ldan = BLKLDD(A, n);
                    CORE_OMP_dgemm(
                        PlasmaConjTrans, PlasmaNoTrans,
                        A.mb, tempmm, A.mb,
                        mzone, A(k, n), ldak,
                               A(k, m), ldak,
                        zone,  A(n, m), ldan);
                }
            }
        }
    }
}
