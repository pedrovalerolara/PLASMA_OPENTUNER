/**
 *
 * @file dcm2ccrb.c
 *
 *  PLASMA computational routine.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 * @generated from compute/zcm2ccrb.c, normal z -> d, Tue Jul 12 11:29:13 2016
 *
 **/

#include "plasma_types.h"
#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_d.h"

/***************************************************************************//**
    @ingroup plasma_cm2ccrb

    Convert column-major (CM) to tiled (CCRB) matrix layout.
    Out-of-place.
*/
void PLASMA_dcm2ccrb_Async(double *Af77, int lda, PLASMA_desc *A,
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
    if (Af77 == NULL) {
        plasma_error("NULL A");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_error("invalid A");
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

    // Check sequence status.
    if (sequence->status != PLASMA_SUCCESS) {
        plasma_request_fail(sequence, request, PLASMA_ERR_SEQUENCE_FLUSHED);
        return;
    }

    // quick return with success
    if (A->m == 0 || A->n == 0)
        return;

    // Call the parallel function.
    plasma_pdoocm2ccrb(Af77, lda, *A, sequence, request);

    return;
}
