/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief Performs panel LU factorization.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * August 15, 2014
 *
 * Modified:
 *   September 30, 2017
 *   May 10, 2019  v7.0.0
 *   December 12, 2021  v7.2.0
 *
 * <pre>
 * Purpose
 * =======
 *   Panel factorization -- block column k
 *
 *   Factor diagonal and subdiagonal blocks and test for exact singularity.
 *   Only the column processes that own block column *k* participate
 *   in the work.
 *
 * Arguments
 * =========
 * options (input) superlu_dist_options_t* (global)
 *         The structure defines the input parameters to control
 *         how the LU decomposition will be performed.
 *
 * k0     (input) int (global)
 *        Counter of the next supernode to be factorized.
 *
 * k      (input) int (global)
 *        The column number of the block column to be factorized.
 *
 * thresh (input) double (global)
 *        The threshold value = s_eps * anorm.
 *
 * Glu_persist (input) Glu_persist_t*
 *        Global data structures (xsup, supno) replicated on all processes.
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh.
 *
 * Llu    (input/output) dLocalLU_t*
 *        Local data structures to store distributed L and U matrices.
 *
 * U_diag_blk_send_req (input/output) MPI_Request*
 *        List of send requests to send down the diagonal block of U.
 *
 * tag_ub (input) int
 *        Upper bound of MPI tag values.
 *
 * stat   (output) SuperLUStat_t*
 *        Record the statistics about the factorization.
 *        See SuperLUStat_t structure defined in util.h.
 *
 * info   (output) int*
 *        = 0: successful exit
 *        < 0: if info = -i, the i-th argument had an illegal value
 *        > 0: if info = i, U(i,i) is exactly zero. The factorization has
 *             been completed, but the factor U is exactly singular,
 *             and division by zero will occur if it is used to solve a
 *             system of equations.
 * </pre>
 */

#include <math.h>
#include "superlu_ddefs.h"
// #include "cblas.h"

/*****************************************************************************
 * The following pdgstrf2_trsm is in version 6 and earlier.
 *****************************************************************************/
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Panel factorization -- block column k
 *
 *   Factor diagonal and subdiagonal blocks and test for exact singularity.
 *   Only the column processes that own block column *k* participate
 *   in the work.
 *
 * Arguments
 * =========
 * options (input) superlu_dist_options_t* (global)
 *         The structure defines the input parameters to control
 *         how the LU decomposition will be performed.
 *
 * k0     (input) int (global)
 *        Counter of the next supernode to be factorized.
 *
 * k      (input) int (global)
 *        The column number of the block column to be factorized.
 *
 * thresh (input) double (global)
 *        The threshold value = s_eps * anorm.
 *
 * Glu_persist (input) Glu_persist_t*
 *        Global data structures (xsup, supno) replicated on all processes.
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh.
 *
 * Llu    (input/output) dLocalLU_t*
 *        Local data structures to store distributed L and U matrices.
 *
 * U_diag_blk_send_req (input/output) MPI_Request*
 *        List of send requests to send down the diagonal block of U.
 *
 * tag_ub (input) int
 *        Upper bound of MPI tag values.
 *
 * stat   (output) SuperLUStat_t*
 *        Record the statistics about the factorization.
 *        See SuperLUStat_t structure defined in util.h.
 *
 * info   (output) int*
 *        = 0: successful exit
 *        < 0: if info = -i, the i-th argument had an illegal value
 *        > 0: if info = i, U(i,i) is exactly zero. The factorization has
 *             been completed, but the factor U is exactly singular,
 *             and division by zero will occur if it is used to solve a
 *             system of equations.
 * </pre>
 */
/* This pdgstrf2 is based on TRSM function */
// (Yida) : This version of this function clean the defines.
// (Yida) : Right-looking LU.
void pdgstrf2_trsm(
    superlu_dist_options_t *options, int_t k0, int_t k, double thresh,
    Glu_persist_t *Glu_persist, gridinfo_t *grid, dLocalLU_t *Llu,
    MPI_Request *U_diag_blk_send_req, int tag_ub,
    SuperLUStat_t *stat, int *info)
{
// printf("[Yida] %s:%d\n", __FILE__, __LINE__);
    int cols_left, iam, l, pkk, pr;
    int incx = 1, incy = 1;

    int nsupr; /* number of rows in the block (LDA) */
    int nsupc; /* number of columns in the block */
    int luptr;
    int_t i, myrow, krow, j, jfst, jlst, u_diag_cnt;
    int_t *xsup = Glu_persist->xsup;
    double *lusup, temp;
    double *ujrow, *ublk_ptr; /* pointer to the U block */
    double alpha = -1, zero = 0.0;
    int_t Pr;
    MPI_Status status;
    MPI_Comm comm = (grid->cscp).comm;
    double t1, t2;

    /* Initialization. */
    iam = grid->iam;
    Pr = grid->nprow;
    myrow = MYROW(iam, grid);
    krow = PROW(k, grid);
    pkk = PNUM(PROW(k, grid), PCOL(k, grid), grid);
    j = LBj(k, grid); /* Local block number */
    jfst = FstBlockC(k);
    jlst = FstBlockC(k + 1);
    lusup = Llu->Lnzval_bc_ptr[j];
    printf("j = %d\n", j);
    nsupc = SuperSize(k);
    if (Llu->Lrowind_bc_ptr[j])
        nsupr = Llu->Lrowind_bc_ptr[j][1];
    else
        nsupr = 0;
    ublk_ptr = ujrow = Llu->ujrow;

    luptr = 0;            /* Point to the diagonal entries. */
    cols_left = nsupc;    /* supernode size */
    int ld_ujrow = nsupc; /* leading dimension of ujrow */
    u_diag_cnt = 0;
    incy = ld_ujrow;

// printf("[Yida] %s:%d\n", __FILE__, __LINE__);
    if (U_diag_blk_send_req &&
        U_diag_blk_send_req[myrow] != MPI_REQUEST_NULL)
    {
printf("[Yida] %s:%d\n", __FILE__, __LINE__);
        /* There are pending sends - wait for all Isend to complete */
        for (pr = 0; pr < Pr; ++pr)
        {
            if (pr != myrow)
            {
                MPI_Wait(U_diag_blk_send_req + pr, &status);
            }
        }
        /* flag no more outstanding send request. */
        U_diag_blk_send_req[myrow] = MPI_REQUEST_NULL;
    }

// printf("[Yida] %s:%d\n", __FILE__, __LINE__);
    if (iam == pkk)
    { /* diagonal process */
// printf("[Yida] %s:%d\n", __FILE__, __LINE__);

        // printf("GETRF Golden.before\n");
        // for(int lyd_i = 0; lyd_i < nsupc * nsupc; lyd_i++){
        //     int lyd_idx = (lyd_i / nsupc) * nsupr + (lyd_i % nsupc);
        //     printf("getrf_block[%d %d] = %le\n", lyd_i, lyd_idx, lusup[lyd_idx]);
        // }
        /* ++++ First step compute diagonal block ++++++++++ */
        for (j = 0; j < jlst - jfst; ++j) // (Yida) : Right-looking LU.
        {                                 /* for each column in panel */
// printf("[Yida] %s:%d\n", __FILE__, __LINE__);
            /* Diagonal pivot */
            i = luptr;
            /* May replace zero pivot.  */
            if (options->ReplaceTinyPivot == YES)
            {
printf("[Yida] %s:%d\n", __FILE__, __LINE__);
                if (fabs(lusup[i]) < thresh)
                { /* Diagonal */
printf("[Yida] %s:%d\n", __FILE__, __LINE__);
                    /* Keep the new diagonal entry with the same sign. */
                    if (lusup[i] < 0)
                        lusup[i] = -thresh;
                    else
                        lusup[i] = thresh;
                    ++(stat->TinyPivots);
                }
            }

            /* storing U in full form  */
            int st;
// printf("[Yida] %s:%d\n", __FILE__, __LINE__);
            for (l = 0; l < cols_left; ++l, i += nsupr, ++u_diag_cnt)
            {
// printf("[Yida] %s:%d\n", __FILE__, __LINE__); // run n times
                st = j * ld_ujrow + j;
                ublk_ptr[st + l * ld_ujrow] = lusup[i]; /* copy one row of U */
            }

// printf("[Yida] %s:%d\n", __FILE__, __LINE__);
            if (ujrow[0] == zero)
            { /* Test for singularity. */
printf("[Yida] %s:%d\n", __FILE__, __LINE__);
                *info = j + jfst + 1;
            }
            else
            { /* Scale the j-th column within diag. block. */
// printf("[Yida] %s:%d\n", __FILE__, __LINE__);
                temp = 1.0 / ujrow[0];
                for (i = luptr + 1; i < luptr - j + nsupc; ++i)
                    lusup[i] *= temp;
                stat->ops[FACT] += nsupc - j - 1;
            }

            /* Rank-1 update of the trailing submatrix within diag. block. */
// printf("[Yida] %s:%d\n", __FILE__, __LINE__);
            if (--cols_left)
            {
                /* l = nsupr - j - 1;  */
                l = nsupc - j - 1; /* Piyush */
// printf("[Yida] %s:%d\n", __FILE__, __LINE__);
                dger_(
                    &l, &cols_left, &alpha,
                    &lusup[luptr + 1], &incx,
                    &ujrow[ld_ujrow], &incy,
                    &lusup[luptr + nsupr + 1], &nsupr); // (Yida) : A -= x * y https://netlib.org/lapack/explore-html/d8/d75/group__ger_gaef5d248da0fdfb62bccb259725935cb8.html
                stat->ops[FACT] += 2 * l * cols_left;
            }

            /* ujrow = ublk_ptr + u_diag_cnt;  */
            ujrow = ujrow + ld_ujrow + 1; /* move to next row of U */
            luptr += nsupr + 1;           /* move to next column */

        } /* for column j ...  first loop */
// printf("[Yida] %s:%d\n", __FILE__, __LINE__);
        // printf("GETRF Golden.after\n");
        // for(int lyd_i = 0; lyd_i < nsupc * nsupc; lyd_i++){
        //     int lyd_idx = (lyd_i / nsupc) * nsupr + (lyd_i % nsupc);
        //     printf("getrf_block[%d] = %le\n", lyd_i, lusup[lyd_idx]);
        // }

        /* ++++ Second step compute off-diagonal block with communication  ++*/
        ublk_ptr = ujrow = Llu->ujrow;
        if (U_diag_blk_send_req && iam == pkk)
        { /* Send the U block downward */
            /** ALWAYS SEND TO ALL OTHERS - TO FIX **/
printf("[Yida] %s:%d\n", __FILE__, __LINE__);
            for (pr = 0; pr < Pr; ++pr)
            {
printf("[Yida] %s:%d\n", __FILE__, __LINE__);
                if (pr != krow)
                {
printf("[Yida] %s:%d\n", __FILE__, __LINE__);
                    /* tag = ((k0<<2)+2) % tag_ub;        */
                    /* tag = (4*(nsupers+k0)+2) % tag_ub; */
                    MPI_Isend(ublk_ptr, nsupc * nsupc, MPI_DOUBLE, pr,
                              SLU_MPI_TAG(4, k0) /* tag */,
                              comm, U_diag_blk_send_req + pr);
                }
            }
            /* flag outstanding Isend */
            U_diag_blk_send_req[krow] = (MPI_Request)TRUE; /* Sherry */
        }

        l = nsupr - nsupc;
        double alpha = 1.0;
printf("[Yida] dtrsm_ %s:%d l=%d nsupc=%d ld_ujrow=%d nsupr=%d\n", __FILE__, __LINE__, l, nsupc, ld_ujrow, nsupr);
        // {
        //     printf("[Yida] Golden.before : \n");
        //     for(int lyd_i = 0; lyd_i < 60; lyd_i++){
        //         int lyd_idx = nsupc + nsupr * (lyd_i / l) + (lyd_i % l);
        //         printf("[%d %d]=%le\n", lyd_i, lyd_idx, lusup[lyd_idx]);
        //     }
        // }
        dtrsm_("R", "U", "N", "N", &l, &nsupc,
               &alpha, ublk_ptr, &ld_ujrow, &lusup[nsupc], &nsupr,
               1, 1, 1, 1);
        // {
        //     printf("[Yida] Golden : \n");
        //     for(int lyd_i = 0; lyd_i < 60; lyd_i++){
        //         int lyd_idx = nsupc + nsupr * (lyd_i / l) + (lyd_i % l);
        //         printf("[%d]=%le\n", lyd_i, lusup[lyd_idx]);
        //     }
        // }
        stat->ops[FACT] += (flops_t)nsupc * (nsupc + 1) * l;
    }
    else
    { /* non-diagonal process */
        /* ================================================================== *
         * Receive the diagonal block of U for panel factorization of L(:,k). *
         * Note: we block for panel factorization of L(:,k), but panel        *
         * factorization of U(:,k) do not block                               *
         * ================================================================== */
printf("[Yida] %s:%d\n", __FILE__, __LINE__);

        /* tag = ((k0<<2)+2) % tag_ub;        */
        /* tag = (4*(nsupers+k0)+2) % tag_ub; */
        // printf("hello message receiving%d %d\n",(nsupc*(nsupc+1))>>1,SLU_MPI_TAG(4,k0));
        MPI_Recv(ublk_ptr, (nsupc * nsupc), MPI_DOUBLE, krow,
                 SLU_MPI_TAG(4, k0) /* tag */,
                 comm, &status);
        if (nsupr > 0)
        {
printf("[Yida] %s:%d\n", __FILE__, __LINE__);
            double alpha = 1.0;
            dtrsm_("R", "U", "N", "N", &nsupr, &nsupc,
                   &alpha, ublk_ptr, &ld_ujrow, lusup, &nsupr, 1, 1, 1, 1);
            stat->ops[FACT] += (flops_t)nsupc * (nsupc + 1) * nsupr;
        }
    } /* end if pkk ... */
printf("[Yida] %s:%d\n", __FILE__, __LINE__);
} /* PDGSTRF2_trsm */

/*****************************************************************************
 * The following functions are for the new pdgstrf2_dtrsm in the 3D code.
 *****************************************************************************/
static int_t LpanelUpdate(int off0, int nsupc, double *ublk_ptr, int ld_ujrow,
                          double *lusup, int nsupr, SCT_t *SCT)
{
    int_t l = nsupr - off0;
    double alpha = 1.0;
    double t1 = SuperLU_timer_();

#define GT 32
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < CEILING(l, GT); ++i)
    {
        int_t off = i * GT;
        int len = SUPERLU_MIN(GT, l - i * GT);

        superlu_dtrsm("R", "U", "N", "N", len, nsupc, alpha,
                      ublk_ptr, ld_ujrow, &lusup[off0 + off], nsupr);

    } /* for i = ... */

    t1 = SuperLU_timer_() - t1;

    SCT->trf2_flops += (double)l * (double)nsupc * (double)nsupc;
    SCT->trf2_time += t1;
    SCT->L_PanelUpdate_tl += t1;
    return 0;
}

#pragma GCC push_options
#pragma GCC optimize("O0")

void dgstrf2(int_t k, double *diagBlk, int_t LDA, double *BlockUfactor, int_t LDU,
             double thresh, int_t *xsup,
             superlu_dist_options_t *options,
             SuperLUStat_t *stat, int *info)
{

    int_t jfst = FstBlockC(k);
    int_t jlst = FstBlockC(k + 1);
    int_t nsupc = SuperSize(k);

    double *ublk_ptr = BlockUfactor;
    double *ujrow = BlockUfactor;
    int_t luptr = 0;       /* Point_t to the diagonal entries. */
    int cols_left = nsupc; /* supernode size */

    for (int_t j = 0; j < nsupc; ++j) /* for each column in panel */
    {
        /* Diagonal pivot */
        int_t i = luptr;
        /* Not to replace zero pivot.  */
        if (options->ReplaceTinyPivot == YES)
        {
            if (fabs(diagBlk[i]) < thresh)
            { /* Diagonal */

#if (PRNTlevel >= 2)
                printf("(%d) .. col %d, tiny pivot %e  ",
                       iam, jfst + j, diagBlk[i]);
#endif
                /* Keep the new diagonal entry with the same sign. */

                if (diagBlk[i] < 0)
                    diagBlk[i] = -thresh;
                else
                    diagBlk[i] = thresh;

#if (PRNTlevel >= 2)
                printf("replaced by %e\n", diagBlk[i]);
#endif
                ++(stat->TinyPivots);
            }
        }

        for (int_t l = 0; l < cols_left; ++l, i += LDA)
        {
            int_t st = j * LDU + j;
            ublk_ptr[st + l * LDU] = diagBlk[i]; /* copy one row of U */
        }
        double alpha = -1, zero = 0.0, one = 1.0;
        if (ujrow[0] == zero) /* Test for singularity. */
        {
            *info = j + jfst + 1;
        }
        else /* Scale the j-th column. */
        {
            double temp;
            temp = 1.0 / ujrow[0];
            for (i = luptr + 1; i < luptr - j + nsupc; ++i)
                diagBlk[i] *= temp;
            stat->ops[FACT] += nsupc - j - 1;
        }

        /* Rank-1 update of the trailing submatrix. */
        if (--cols_left)
        {
            /*following must be int*/
            int l = nsupc - j - 1;
            int incx = 1;
            int incy = LDU;
            /* Rank-1 update */
            superlu_dger(l, cols_left, alpha, &diagBlk[luptr + 1], incx,
                         &ujrow[LDU], incy, &diagBlk[luptr + LDA + 1],
                         LDA);
            stat->ops[FACT] += 2 * l * cols_left;
        }

        ujrow = ujrow + LDU + 1; /* move to next row of U */
        luptr += LDA + 1;        /* move to next column */

    } /* for column j ...  first loop */

    // printf("Coming to local dgstrf2\n");
}

/************************************************************************/
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Factorize the diagonal block; called from process that owns the (k,k) block
 *
 * Arguments
 * =========
 *
 * info   (output) int*
 *        = 0: successful exit
 *        > 0: if info = i, U(i,i) is exactly zero. The factorization has
 *             been completed, but the factor U is exactly singular,
 *             and division by zero will occur if it is used to solve a
 *             system of equations.
 */
void Local_Dgstrf2(superlu_dist_options_t *options, int_t k, double thresh,
                   double *BlockUFactor, /*factored U is overwritten here*/
                   Glu_persist_t *Glu_persist, gridinfo_t *grid, dLocalLU_t *Llu,
                   SuperLUStat_t *stat, int *info, SCT_t *SCT)
{
    // double t1 = SuperLU_timer_();
    int_t *xsup = Glu_persist->xsup;
    double alpha = -1, zero = 0.0;

    // printf("Entering dgetrf2 %d \n", k);
    /* Initialization. */
    int_t lk = LBj(k, grid); /* Local block number */
    int_t jfst = FstBlockC(k);
    int_t jlst = FstBlockC(k + 1);
    double *lusup = Llu->Lnzval_bc_ptr[lk];
    int nsupc = SuperSize(k);
    int nsupr;
    if (Llu->Lrowind_bc_ptr[lk])
        nsupr = Llu->Lrowind_bc_ptr[lk][1];
    else
        nsupr = 0;
    double *ublk_ptr = BlockUFactor;
    double *ujrow = BlockUFactor;
    int_t luptr = 0;       /* Point_t to the diagonal entries. */
    int cols_left = nsupc; /* supernode size */
    int_t u_diag_cnt = 0;
    int_t ld_ujrow = nsupc; /* leading dimension of ujrow */
    int incx = 1;
    int incy = ld_ujrow;

    for (int_t j = 0; j < jlst - jfst; ++j) /* for each column in panel */
    {
        /* Diagonal pivot */
        int_t i = luptr;
        /* Allow to replace zero pivot.  */
        // if (options->ReplaceTinyPivot == YES && lusup[i] != 0.0)
        if (options->ReplaceTinyPivot == YES)
        {
            if (fabs(lusup[i]) < thresh)
            { /* Diagonal */

#if (PRNTlevel >= 2)
                printf("(%d) .. col %d, tiny pivot %e  ",
                       iam, jfst + j, lusup[i]);
#endif
                /* Keep the new diagonal entry with the same sign. */
                if (lusup[i] < 0)
                    lusup[i] = -thresh;
                else
                    lusup[i] = thresh;
#if (PRNTlevel >= 2)
                printf("replaced by %e\n", lusup[i]);
#endif
                ++(stat->TinyPivots);
            }
        }

        for (int_t l = 0; l < cols_left; ++l, i += nsupr, ++u_diag_cnt)
        {
            int_t st = j * ld_ujrow + j;
            ublk_ptr[st + l * ld_ujrow] = lusup[i]; /* copy one row of U */
        }

        if (ujrow[0] == zero) /* Test for singularity. */
        {
            *info = j + jfst + 1;
        }
        else /* Scale the j-th column. */
        {
            double temp;
            temp = 1.0 / ujrow[0];
            for (int_t i = luptr + 1; i < luptr - j + nsupc; ++i)
                lusup[i] *= temp;
            stat->ops[FACT] += nsupc - j - 1;
        }

        /* Rank-1 update of the trailing submatrix. */
        if (--cols_left)
        {
            /*following must be int*/
            int l = nsupc - j - 1;

            /* Rank-1 update */
            superlu_dger(l, cols_left, alpha, &lusup[luptr + 1], incx,
                         &ujrow[ld_ujrow], incy, &lusup[luptr + nsupr + 1], nsupr);
            stat->ops[FACT] += 2 * l * cols_left;
        }

        ujrow = ujrow + ld_ujrow + 1; /* move to next row of U */
        luptr += nsupr + 1;           /* move to next column */

    } /* for column j ...  first loop */

    // int_t thread_id = omp_get_thread_num();
    //  SCT->Local_Dgstrf2_Thread_tl[thread_id * CACHE_LINE_SIZE] += (double) ( SuperLU_timer_() - t1);
} /* end Local_Dgstrf2 */

#pragma GCC pop_options
/************************************************************************/
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Panel factorization -- block column k
 *
 *   Factor diagonal and subdiagonal blocks and test for exact singularity.
 *   Only the column processes that own block column *k* participate
 *   in the work.
 *
 * Arguments
 * =========
 * options (input) superlu_dist_options_t* (global)
 *         The structure defines the input parameters to control
 *         how the LU decomposition will be performed.
 *
 * nsupers (input) int_t (global)
 *         Number of supernodes.
 *
 * k0     (input) int (global)
 *        Counter of the next supernode to be factorized.
 *
 * k      (input) int (global)
 *        The column number of the block column to be factorized.
 *
 * thresh (input) double (global)
 *        The threshold value = s_eps * anorm.
 *
 * Glu_persist (input) Glu_persist_t*
 *        Global data structures (xsup, supno) replicated on all processes.
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh.
 *
 * Llu    (input/output) dLocalLU_t*
 *        Local data structures to store distributed L and U matrices.
 *
 * U_diag_blk_send_req (input/output) MPI_Request*
 *        List of send requests to send down the diagonal block of U.
 *
 * tag_ub (input) int
 *        Upper bound of MPI tag values.
 *
 * stat   (output) SuperLUStat_t*
 *        Record the statistics about the factorization.
 *        See SuperLUStat_t structure defined in util.h.
 *
 * info   (output) int*
 *        = 0: successful exit
 *        < 0: if info = -i, the i-th argument had an illegal value
 *        > 0: if info = i, U(i,i) is exactly zero. The factorization has
 *             been completed, but the factor U is exactly singular,
 *             and division by zero will occur if it is used to solve a
 *             system of equations.
 *
 * SCT    (output) SCT_t*
 *        Additional statistics used in the 3D algorithm.
 *
 * </pre>
 */
void pdgstrf2_xtrsm(superlu_dist_options_t *options, int_t nsupers,
                    int_t k0, int_t k, double thresh, Glu_persist_t *Glu_persist,
                    gridinfo_t *grid, dLocalLU_t *Llu, MPI_Request *U_diag_blk_send_req,
                    int tag_ub, SuperLUStat_t *stat, int *info, SCT_t *SCT)
{
    int cols_left, iam, pkk;
    int incy = 1;

    int nsupr; /* number of rows in the block (LDA) */
    int luptr;
    int_t myrow, krow, j, jfst, jlst, u_diag_cnt;
    int_t nsupc; /* number of columns in the block */
    int_t *xsup = Glu_persist->xsup;
    double *lusup;
    double *ujrow, *ublk_ptr; /* pointer to the U block */
    int_t Pr;

    /* Quick return. */
    *info = 0;

    /* Initialization. */
    iam = grid->iam;
    Pr = grid->nprow;
    myrow = MYROW(iam, grid);
    krow = PROW(k, grid);
    pkk = PNUM(PROW(k, grid), PCOL(k, grid), grid);
    j = LBj(k, grid); /* Local block number */
    jfst = FstBlockC(k);
    jlst = FstBlockC(k + 1);
    lusup = Llu->Lnzval_bc_ptr[j];
    nsupc = SuperSize(k);
    if (Llu->Lrowind_bc_ptr[j])
        nsupr = Llu->Lrowind_bc_ptr[j][1];
    else
        nsupr = 0;
    ublk_ptr = ujrow = Llu->ujrow;

    luptr = 0;            /* Point to the diagonal entries. */
    cols_left = nsupc;    /* supernode size */
    int ld_ujrow = nsupc; /* leading dimension of ujrow */
    u_diag_cnt = 0;
    incy = ld_ujrow;

    if (U_diag_blk_send_req && U_diag_blk_send_req[myrow])
    {
        /* There are pending sends - wait for all Isend to complete */
        Wait_UDiagBlockSend(U_diag_blk_send_req, grid, SCT);
    }

    if (iam == pkk) /* diagonal process */
    {
        /*factorize the diagonal block*/
        Local_Dgstrf2(options, k, thresh, Llu->ujrow, Glu_persist,
                      grid, Llu, stat, info, SCT);
        ublk_ptr = ujrow = Llu->ujrow;

        if (U_diag_blk_send_req && iam == pkk) /* Send the U block */
        {
            dISend_UDiagBlock(k0, ublk_ptr, nsupc * nsupc, U_diag_blk_send_req,
                              grid, tag_ub);
            U_diag_blk_send_req[krow] = (MPI_Request)TRUE; /* flag outstanding Isend */
        }

        LpanelUpdate(nsupc, nsupc, ublk_ptr, ld_ujrow, lusup, nsupr, SCT);
    }
    else /* non-diagonal process */
    {
        /* ================================================ *
         * Receive the diagonal block of U                  *
         * for panel factorization of L(:,k)                *
         * note: we block for panel factorization of L(:,k) *
         * but panel factorization of U(:,k) don't          *
         * ================================================ */

        dRecv_UDiagBlock(k0, ublk_ptr, (nsupc * nsupc), krow, grid, SCT, tag_ub);

        if (nsupr > 0)
        {
            LpanelUpdate(0, nsupc, ublk_ptr, ld_ujrow, lusup, nsupr, SCT);
        }
    } /* end if pkk ... */

} /* pdgstrf2_xtrsm */

/*****************************************************************************
 * The following functions are for the new pdgstrs2_omp in the 3D code.
 *****************************************************************************/

/* PDGSTRS2 helping kernels*/

int_t dTrs2_GatherU(int_t iukp, int_t rukp, int_t klst,
                    int_t nsupc, int_t ldu,
                    int_t *usub,
                    double *uval, double *tempv)
{
    double zero = 0.0;
    int_t ncols = 0;
    for (int_t jj = iukp; jj < iukp + nsupc; ++jj)
    {
        int_t segsize = klst - usub[jj];
        if (segsize)
        {
            int_t lead_zero = ldu - segsize;
            for (int_t i = 0; i < lead_zero; ++i)
                tempv[i] = zero;
            tempv += lead_zero;
            for (int_t i = 0; i < segsize; ++i)
                tempv[i] = uval[rukp + i];
            rukp += segsize;
            tempv += segsize;
            ncols++;
        }
    }
    return ncols;
}

int_t dTrs2_ScatterU(int_t iukp, int_t rukp, int_t klst,
                     int_t nsupc, int_t ldu,
                     int_t *usub, double *uval, double *tempv)
{
    for (int_t jj = 0; jj < nsupc; ++jj)
    {
        int_t segsize = klst - usub[iukp + jj];
        if (segsize)
        {
            int_t lead_zero = ldu - segsize;
            tempv += lead_zero;
            for (int i = 0; i < segsize; ++i)
            {
                uval[rukp + i] = tempv[i];
            }
            tempv += segsize;
            rukp += segsize;
        }
    } /*for jj=0:nsupc */
    return 0;
}

int_t dTrs2_GatherTrsmScatter(int_t klst, int_t iukp, int_t rukp,
                              int_t *usub, double *uval, double *tempv,
                              int_t knsupc, int nsupr, double *lusup,
                              Glu_persist_t *Glu_persist) /*glupersist for xsup for supersize*/
{
    double alpha = 1.0;
    int_t *xsup = Glu_persist->xsup;
    // int_t iukp = Ublock_info.iukp;
    // int_t rukp = Ublock_info.rukp;
    int_t gb = usub[iukp];
    int_t nsupc = SuperSize(gb);
    iukp += UB_DESCRIPTOR;

    // printf("klst inside task%d\n", );
    /*find ldu */
    int ldu = 0;
    for (int_t jj = iukp; jj < iukp + nsupc; ++jj)
    {
        ldu = SUPERLU_MAX(klst - usub[jj], ldu);
    }

    /*pack U block into a dense Block*/
    int ncols = dTrs2_GatherU(iukp, rukp, klst, nsupc, ldu, usub,
                              uval, tempv);

    /*now call dtrsm on packed dense block*/
    int_t luptr = (knsupc - ldu) * (nsupr + 1);
    // if(ldu>nsupr) printf("nsupr %d ldu %d\n",nsupr,ldu );

    superlu_dtrsm("L", "L", "N", "U", ldu, ncols, alpha,
                  &lusup[luptr], nsupr, tempv, ldu);

    /*now scatter the output into sparse U block*/
    dTrs2_ScatterU(iukp, rukp, klst, nsupc, ldu, usub, uval, tempv);

    return 0;
}
/* END 3D CODE */
/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

/*****************************************************************************
 * The following pdgstrf2_omp is improved for KNL, since Version 5.2.0.
 *****************************************************************************/
void pdgstrs2_omp(
    int_t k0,
    int_t k,
    Glu_persist_t *Glu_persist,
    gridinfo_t *grid,
    dLocalLU_t *Llu,
    Ublock_info_t *Ublock_info,
    SuperLUStat_t *stat)
{
printf("[Yida] %s:%d\n", __FILE__, __LINE__);
    int iam, pkk;
    int incx = 1;
    int nsupr; /* number of rows in the block L(:,k) (LDA) */
    int segsize;
    int nsupc; /* number of columns in the block */
    int_t iukp, rukp;
    int_t b, gb, j, klst, knsupc, lk, nb;
    int_t *xsup = Glu_persist->xsup;
    int_t *usub;
    double *lusup, *uval;

    /* Quick return. */
    lk = LBi(k, grid); /* Local block number */
    if (!Llu->Unzval_br_ptr[lk])
        return;

    /* Initialization. */
    iam = grid->iam;
    pkk = PNUM(PROW(k, grid), PCOL(k, grid), grid);
    klst = FstBlockC(k + 1);
    knsupc = SuperSize(k);
    usub = Llu->Ufstnz_br_ptr[lk]; /* index[] of block row U(k,:) */
    uval = Llu->Unzval_br_ptr[lk];
printf("[Yida] %s:%d lk=%d\n", __FILE__, __LINE__, lk);
    if (iam == pkk)
    {
printf("[Yida] %s:%d\n", __FILE__, __LINE__);
        lk = LBj(k, grid);
        nsupr = Llu->Lrowind_bc_ptr[lk][1]; /* LDA of lusup[] */
        lusup = Llu->Lnzval_bc_ptr[lk];
    }
    else
    {
printf("[Yida] %s:%d\n", __FILE__, __LINE__);
        nsupr = Llu->Lsub_buf_2[k0 % (1 + stat->num_look_aheads)][1]; /* LDA of lusup[] */
        lusup = Llu->Lval_buf_2[k0 % (1 + stat->num_look_aheads)];
    }

    /////////////////////new-test//////////////////////////
    /* !! Taken from Carl/SuperLU_DIST_5.1.0/EXAMPLE/pdgstrf2_v3.c !! */

    /* Master thread: set up pointers to each block in the row */
    nb = usub[0];
    iukp = BR_HEADER;
    rukp = 0;

    /* Sherry: can use the existing  Ublock_info[] array, call
       Trs2_InitUblock_info();                                 */
#undef USE_Ublock_info
    int *blocks_index_pointers = SUPERLU_MALLOC(3 * nb * sizeof(int));
    int *blocks_value_pointers = blocks_index_pointers + nb;
    int *nsupc_temp = blocks_value_pointers + nb;
printf("[Yida] %s:%d\n", __FILE__, __LINE__);
    for (b = 0; b < nb; b++)
    { /* set up pointers to each block */
printf("[Yida] %s:%d\n", __FILE__, __LINE__);
        blocks_index_pointers[b] = iukp + UB_DESCRIPTOR;
        blocks_value_pointers[b] = rukp;
        gb = usub[iukp];
        rukp += usub[iukp + 1];
        nsupc = SuperSize(gb);
        nsupc_temp[b] = nsupc;
        iukp += (UB_DESCRIPTOR + nsupc); /* move to the next block */
    }

// Sherry: this version is more NUMA friendly compared to pdgstrf2_v2.c
// https://stackoverflow.com/questions/13065943/task-based-programming-pragma-omp-task-versus-pragma-omp-parallel-for
/* Loop through all the blocks in the row. */
printf("[Yida] %s:%d\n", __FILE__, __LINE__);
#pragma omp parallel for schedule(static) default(shared) private(b, j, iukp, rukp, segsize)
    for (b = 0; b < nb; ++b)
    {
printf("[Yida] %s:%d\n", __FILE__, __LINE__);
        iukp = blocks_index_pointers[b];
        rukp = blocks_value_pointers[b];

        /* Loop through all the segments in the block. */
        for (j = 0; j < nsupc_temp[b]; j++)
        {
printf("[Yida] %s:%d\n", __FILE__, __LINE__);
            segsize = klst - usub[iukp++];
            if (segsize)
            {
printf("[Yida] %s:%d\n", __FILE__, __LINE__);
#pragma omp task default(shared) firstprivate(segsize, rukp) if (segsize > 30)
                { /* Nonzero segment. */
                    int_t luptr = (knsupc - segsize) * (nsupr + 1);
printf("[Yida] %s:%d rukp=%d dtrsv_(n=%d, lda=%d, incx=%d)\n", __FILE__, __LINE__, rukp, segsize, nsupr, incx);
                    dtrsv_("L", "N", "U", &segsize, &lusup[luptr], &nsupr,
                           &uval[rukp], &incx, 1, 1, 1);
                } /* end task */
                rukp += segsize;
                stat->ops[FACT] += segsize * (segsize + 1);
            } /* end if segsize > 0 */
        } /* end for j in parallel ... */
    } /* end for b ... */
printf("[Yida] %s:%d\n", __FILE__, __LINE__);

    /* Deallocate memory */
    SUPERLU_FREE(blocks_index_pointers);
} /* pdgstrs2_omp */
