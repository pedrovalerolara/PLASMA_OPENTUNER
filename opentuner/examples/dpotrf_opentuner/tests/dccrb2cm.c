/**
 *
 * @file dccrb2cm.c
 *
 *  PLASMA computational routine.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 3.0.0
 * @author Jakub Kurzak
 * @date 2016-01-01
 * @generated from compute/zccrb2cm.c, normal z -> d, Tue Jul 12 12:29:59 2016
 *
 **/

#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_d.h"
#include "plasma_types.h"

/***************************************************************************//**
    @ingroup plasma_ccrb2cm

    Convert tiled (CCRB) to column-major (CM) matrix layout.
    Out-of-place.
*/
void PLASMA_dccrb2cm_Async(PLASMA_desc *A, double *Af77, int lda,
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
    if (plasma_desc_check(A) != PLASMA_SUCCESS) {
        plasma_error("invalid A");
        plasma_request_fail(sequence, request, PLASMA_ERR_ILLEGAL_VALUE);
        return;
    }
    if (Af77 == NULL) {
        plasma_error("NULL A");
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
    plasma_pdooccrb2cm(*A, Af77, lda, sequence, request);

    return;
}
