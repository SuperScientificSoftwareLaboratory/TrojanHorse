/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
#include <math.h>
#include "superlu_ddefs.h"
#include "gpu_api_utils.h"
#define PHI_FRAMEWORK
#define CACHELINE 0 /* not worry about false sharing of different threads */
#define GEMM_PADLEN 8
#define PDGSTRF2 pdgstrf2_trsm
extern void isort(int_t N, int_t *ARRAY1, int_t *ARRAY2);
extern void isort1(int_t N, int_t *ARRAY);
#include "dscatter.c"

int_t pdgstrf(superlu_dist_options_t *options, int m, int n, double anorm,
              dLUstruct_t *LUstruct, gridinfo_t *grid, SuperLUStat_t *stat, int *info)
{
    double zero = 0.0, alpha = 1.0, beta = 0.0;
    int_t *xsup;
    int_t *lsub, *lsub1, *usub, *Usub_buf;
    int_t **Lsub_buf_2, **Usub_buf_2;
    double **Lval_buf_2, **Uval_buf_2;        /* pointers to starts of bufs */
    double *lusup, *lusup1, *uval, *Uval_buf; /* pointer to current buf     */
    int_t fnz, i, ib, ijb, ilst, it, iukp, jj, klst,
        ldv, lptr, lptr0, lptrj, luptr, luptr0, luptrj,
        nlb, nub, rel, rukp, il, iu;
    int jb, ljb, nsupc, knsupc, lb, lib;
    int Pc, Pr;
    int iam, kcol, krow, yourcol, mycol, myrow, pi, pj;
    int j, k, lk, nsupers;      /* k - current panel to work on */
    int k0;                     /* counter of the next supernode to be factored */
    int kk, kk0, kk1, kk2, jj0; /* panels in the look-ahead window */
    int iukp0, rukp0, flag0, flag1;
    int nsupr, nbrow, segsize;
    int msg0, msg2;
    int_t **Ufstnz_br_ptr, **Lrowind_bc_ptr;
    double **Unzval_br_ptr, **Lnzval_bc_ptr;
    int_t *index;
    double *nzval;
    double *ucol;
    int *indirect, *indirect2;
    int_t *tempi;
    double *tempu, *tempv, *tempr;
    /*    double *tempv2d, *tempU2d;  Sherry */
    int iinfo;
    int *ToRecv, *ToSendD, **ToSendR;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    superlu_scope_t *scp;
    float s_eps;
    double thresh;
    /*int full;*/
    int ldt, ldu, lead_zero, ncols, ncb, nrb, p, pr, pc, nblocks;
    int_t *etree_supno_l, *etree_supno, *blocks, *blockr, *Ublock, *Urows,
        *Lblock, *Lrows, *perm_u, *sf_block, *sf_block_l, *nnodes_l,
        *nnodes_u, *edag_supno_l, *recvbuf, **edag_supno;
    float edag_supno_l_bytes;
    int_t *iperm_u;
    int *msgcnt;               /* Count the size of the message xfer'd in each buffer:
                                *     0 : transferred in Lsub_buf[]
                                *     1 : transferred in Lval_buf[]
                                *     2 : transferred in Usub_buf[]
                                *     3 : transferred in Uval_buf[]
                                */
    int **msgcnts, **msgcntsU; /* counts in the look-ahead window */
    int *factored;             /* factored[j] == 0 : L col panel j is factorized. */
    int *factoredU;            /* factoredU[i] == 1 : U row panel i is factorized. */
    int nnodes, *sendcnts, *sdispls, *recvcnts, *rdispls, *srows, *rrows;
    etree_node *head, *tail, *ptr;
    int *num_child;
    int num_look_aheads, look_id;
    int *look_ahead; /* global look_ahead table */
    int_t *perm_c_supno, *iperm_c_supno;
    /* perm_c_supno[k] = j means at the k-th step of elimination,
     * the j-th supernode is chosen. */
    MPI_Request *recv_req, **recv_reqs, **send_reqs, **send_reqs_u,
        **recv_reqs_u;
    MPI_Request *send_req, *U_diag_blk_send_req = NULL;
    MPI_Status status;
    void *attr_val;
    int flag;

    /* The following variables are used to pad GEMM dimensions so that
       each is a multiple of vector length (8 doubles for KNL)  */
    int gemm_m_pad = GEMM_PADLEN, gemm_k_pad = GEMM_PADLEN,
        gemm_n_pad = GEMM_PADLEN;
    int gemm_padding = 0;

    int iword = sizeof(int_t);
    int dword = sizeof(double);

    /* For measuring load imbalence in omp threads */
    double omp_load_imblc = 0.0;
    double *omp_loop_time;

    double schur_flop_timer = 0.0;
    double pdgstrf2_timer = 0.0;
    double pdgstrs2_timer = 0.0;
    double lookaheadupdatetimer = 0.0;
    double InitTimer = 0.0; /* including compute schedule, malloc */
    double tt_start, tt_end;

    /* Counters for memory operations and timings */
    double scatter_mem_op_counter = 0.0;
    double scatter_mem_op_timer = 0.0;
    double scatterL_mem_op_counter = 0.0;
    double scatterL_mem_op_timer = 0.0;
    double scatterU_mem_op_counter = 0.0;
    double scatterU_mem_op_timer = 0.0;

    /* Counters for flops/gather/scatter and timings */
    double GatherLTimer = 0.0;
    double LookAheadRowSepMOP = 0.0;
    double GatherUTimer = 0.0;
    double GatherMOP = 0.0;
    double LookAheadGEMMTimer = 0.0;
    double LookAheadGEMMFlOp = 0.0;
    double LookAheadScatterTimer = 0.0;
    double LookAheadScatterMOP = 0.0;
    double RemainGEMMTimer = 0.0;
    double RemainGEMM_flops = 0.0;
    double RemainScatterTimer = 0.0;
    double NetSchurUpTimer = 0.0;
    double schur_flop_counter = 0.0;

    /* Test the input parameters. */
    *info = 0;
    if (m < 0)
        *info = -2;
    else if (n < 0)
        *info = -3;
    if (*info)
    {
        pxerr_dist("pdgstrf", grid, -*info);
        return (-1);
    }

    /* Quick return if possible. */
    if (m == 0 || n == 0)
        return 0;

    double tt1 = SuperLU_timer_();

    /*
     * Initialization.
     */
    iam = grid->iam;
    Pc = grid->npcol;
    Pr = grid->nprow;
    myrow = MYROW(iam, grid);
    mycol = MYCOL(iam, grid);
    nsupers = Glu_persist->supno[n - 1] + 1;
    xsup = Glu_persist->xsup;
    s_eps = smach_dist("Epsilon");
    thresh = s_eps * anorm;

    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &attr_val, &flag);
    if (!flag)
    {
        fprintf(stderr, "Could not get TAG_UB\n");
        return (-1);
    }
    int tag_ub = *(int *)attr_val;

    stat->ops[FACT] = 0.0;
    stat->current_buffer = 0.0;
    stat->peak_buffer = 0.0;
    stat->gpu_buffer = 0.0;

    /* make sure the range of look-ahead window [0, MAX_LOOKAHEADS-1] */
    num_look_aheads = SUPERLU_MAX(0, SUPERLU_MIN(options->num_lookaheads, MAX_LOOKAHEADS - 1));

    if (Pr * Pc > 1)
    {
        if (!(U_diag_blk_send_req =
                  (MPI_Request *)SUPERLU_MALLOC(Pr * sizeof(MPI_Request))))
            ABORT("Malloc fails for U_diag_blk_send_req[].");
        /* flag no outstanding Isend */
        U_diag_blk_send_req[myrow] = MPI_REQUEST_NULL; /* used 0 before */

        /* allocating buffers for look-ahead */
        i = Llu->bufmax[0];
        if (i != 0)
        {
            if (!(Llu->Lsub_buf_2[0] = intMalloc_dist((num_look_aheads + 1) * ((size_t)i))))
                ABORT("Malloc fails for Lsub_buf.");
            tempi = Llu->Lsub_buf_2[0];
            for (jj = 0; jj < num_look_aheads; jj++)
                Llu->Lsub_buf_2[jj + 1] = tempi + i * (jj + 1); /* vectorize */
                                                                // Llu->Lsub_buf_2[jj + 1] = Llu->Lsub_buf_2[jj] + i;
        }
        i = Llu->bufmax[1];
        if (i != 0)
        {
            if (!(Llu->Lval_buf_2[0] = doubleMalloc_dist((num_look_aheads + 1) * ((size_t)i))))
                ABORT("Malloc fails for Lval_buf[].");
            tempr = Llu->Lval_buf_2[0];
            for (jj = 0; jj < num_look_aheads; jj++)
                Llu->Lval_buf_2[jj + 1] = tempr + i * (jj + 1); /* vectorize */
                                                                // Llu->Lval_buf_2[jj + 1] = Llu->Lval_buf_2[jj] + i;
        }
        i = Llu->bufmax[2];
        if (i != 0)
        {
            if (!(Llu->Usub_buf_2[0] = intMalloc_dist((num_look_aheads + 1) * i)))
                ABORT("Malloc fails for Usub_buf_2[].");
            tempi = Llu->Usub_buf_2[0];
            for (jj = 0; jj < num_look_aheads; jj++)
                Llu->Usub_buf_2[jj + 1] = tempi + i * (jj + 1); /* vectorize */
                                                                // Llu->Usub_buf_2[jj + 1] = Llu->Usub_buf_2[jj] + i;
        }
        i = Llu->bufmax[3];
        if (i != 0)
        {
            if (!(Llu->Uval_buf_2[0] = doubleMalloc_dist((num_look_aheads + 1) * i)))
                ABORT("Malloc fails for Uval_buf_2[].");
            tempr = Llu->Uval_buf_2[0];
            for (jj = 0; jj < num_look_aheads; jj++)
                Llu->Uval_buf_2[jj + 1] = tempr + i * (jj + 1); /* vectorize */
                                                                // Llu->Uval_buf_2[jj + 1] = Llu->Uval_buf_2[jj] + i;
        }
    }

    log_memory((Llu->bufmax[0] + Llu->bufmax[2]) * (num_look_aheads + 1) * iword +
                   (Llu->bufmax[1] + Llu->bufmax[3]) * (num_look_aheads + 1) * dword,
               stat);

    /* creating pointers to the look-ahead buffers */
    if (!(Lsub_buf_2 = SUPERLU_MALLOC((1 + num_look_aheads) * sizeof(int_t *))))
        ABORT("Malloc fails for Lsub_buf_2[].");
    if (!(Lval_buf_2 = SUPERLU_MALLOC((1 + num_look_aheads) * sizeof(double *))))
        ABORT("Malloc fails for Lval_buf_2[].");
    if (!(Usub_buf_2 = SUPERLU_MALLOC((1 + num_look_aheads) * sizeof(int_t *))))
        ABORT("Malloc fails for Uval_buf_2[].");
    if (!(Uval_buf_2 = SUPERLU_MALLOC((1 + num_look_aheads) * sizeof(double *))))
        ABORT("Malloc fails for buf_2[].");
    for (i = 0; i <= num_look_aheads; i++)
    {
        Lval_buf_2[i] = Llu->Lval_buf_2[i];
        Lsub_buf_2[i] = Llu->Lsub_buf_2[i];
        Uval_buf_2[i] = Llu->Uval_buf_2[i];
        Usub_buf_2[i] = Llu->Usub_buf_2[i];
    }

    if (!(msgcnts = SUPERLU_MALLOC((1 + num_look_aheads) * sizeof(int *))))
        ABORT("Malloc fails for msgcnts[].");
    if (!(msgcntsU = SUPERLU_MALLOC((1 + num_look_aheads) * sizeof(int *))))
        ABORT("Malloc fails for msgcntsU[].");
    for (i = 0; i <= num_look_aheads; i++)
    {
        if (!(msgcnts[i] = SUPERLU_MALLOC(4 * sizeof(int))))
            ABORT("Malloc fails for msgcnts[].");
        if (!(msgcntsU[i] = SUPERLU_MALLOC(4 * sizeof(int))))
            ABORT("Malloc fails for msgcntsU[].");
    }

    if (!(recv_reqs_u = SUPERLU_MALLOC((1 + num_look_aheads) * sizeof(MPI_Request *))))
        ABORT("Malloc fails for recv_reqs_u[].");
    if (!(send_reqs_u = SUPERLU_MALLOC((1 + num_look_aheads) * sizeof(MPI_Request *))))
        ABORT("Malloc fails for send_reqs_u[].");
    if (!(send_reqs = SUPERLU_MALLOC((1 + num_look_aheads) * sizeof(MPI_Request *))))
        ABORT("Malloc fails for send_reqs_u[].");
    if (!(recv_reqs = SUPERLU_MALLOC((1 + num_look_aheads) * sizeof(MPI_Request *))))
        ABORT("Malloc fails for recv_reqs[].");
    for (i = 0; i <= num_look_aheads; i++)
    {
        if (!(recv_reqs_u[i] = (MPI_Request *)SUPERLU_MALLOC(2 * sizeof(MPI_Request))))
            ABORT("Malloc fails for recv_req_u[i].");
        if (!(send_reqs_u[i] = (MPI_Request *)SUPERLU_MALLOC(2 * Pr * sizeof(MPI_Request))))
            ABORT("Malloc fails for send_req_u[i].");
        if (!(send_reqs[i] = (MPI_Request *)SUPERLU_MALLOC(2 * Pc * sizeof(MPI_Request))))
            ABORT("Malloc fails for send_reqs[i].");
        if (!(recv_reqs[i] = (MPI_Request *)SUPERLU_MALLOC(4 * sizeof(MPI_Request))))
            ABORT("Malloc fails for recv_req[].");
        send_reqs[i][0] = send_reqs[i][1] = MPI_REQUEST_NULL;
        recv_reqs[i][0] = recv_reqs[i][1] = MPI_REQUEST_NULL;
    }

    if (!(factored = SUPERLU_MALLOC(nsupers * sizeof(int))))
        ABORT("Malloc fails for factored[].");
    if (!(factoredU = SUPERLU_MALLOC(nsupers * sizeof(int))))
        ABORT("Malloc fails for factoredU[].");
    for (i = 0; i < nsupers; i++)
        factored[i] = factoredU[i] = -1;

    log_memory(2 * nsupers * iword, stat);

    int num_threads = 1;

#pragma omp parallel default(shared)
#pragma omp master
    {
        num_threads = omp_get_num_threads();
    }

    omp_loop_time = (double *)SUPERLU_MALLOC(num_threads * sizeof(double));

    nblocks = 0;
    ncb = nsupers / Pc; /* number of column blocks, horizontal */ // (Yida) : max number of supercol one process would process
    nrb = nsupers / Pr; /* number of row blocks, vertical  */     // (Yida) : max number of superrow one process would process

    /* in order to have dynamic scheduling */
    int *full_u_cols;
    int *blk_ldu;
    full_u_cols = SUPERLU_MALLOC((ncb + 1) * sizeof(int));
    blk_ldu = SUPERLU_MALLOC((ncb + 1) * sizeof(int)); // +1 to accommodate un-even division

    log_memory(2 * ncb * iword, stat);

    /* ##################################################################
     *  Compute a good static schedule based on the factorization task graph.
     * ################################################################## */
    perm_c_supno = SUPERLU_MALLOC(2 * nsupers * sizeof(int_t));
    iperm_c_supno = perm_c_supno + nsupers;

    dstatic_schedule(options, m, n, LUstruct, grid, stat,
                     perm_c_supno, iperm_c_supno, info);

    /* ################################################################## */

    /* constructing look-ahead table to indicate the last dependency */
    int *look_ahead_l; /* Sherry: add comment on look_ahead_l[] */
    stat->num_look_aheads = num_look_aheads;

    look_ahead_l = SUPERLU_MALLOC(nsupers * sizeof(int));
    look_ahead = SUPERLU_MALLOC(nsupers * sizeof(int));
    for (lb = 0; lb < nsupers; lb++)
        look_ahead_l[lb] = -1; /* vectorized */
    log_memory(3 * nsupers * iword, stat);

    /* Sherry: omp parallel?
       not worth doing, due to concurrent write to look_ahead_l[jb] */
    for (lb = 0; lb < nrb; ++lb)
    { /* go through U-factor */
        ib = lb * Pr + myrow;
        index = Llu->Ufstnz_br_ptr[lb];
        if (index)
        { /* Not an empty row */
            k = BR_HEADER;
            for (j = 0; j < index[0]; ++j)
            {
                jb = index[k]; /* global block number */
                if (jb != ib)
                    look_ahead_l[jb] =
                        SUPERLU_MAX(iperm_c_supno[ib], look_ahead_l[jb]);
                k += UB_DESCRIPTOR + SuperSize(index[k]);
            }
        }
    }
    if (myrow < nsupers % grid->nprow)
    { /* leftover block rows */
        ib = nrb * Pr + myrow;
        index = Llu->Ufstnz_br_ptr[nrb];
        if (index)
        { /* Not an empty row */
            k = BR_HEADER;
            for (j = 0; j < index[0]; ++j)
            {
                jb = index[k];
                if (jb != ib)
                    look_ahead_l[jb] =
                        SUPERLU_MAX(iperm_c_supno[ib], look_ahead_l[jb]);
                k += UB_DESCRIPTOR + SuperSize(index[k]);
            }
        }
    }

    if (options->SymPattern == NO)
    {
        /* Sherry: omp parallel?
           not worth doing, due to concurrent write to look_ahead_l[jb] */
        for (lb = 0; lb < ncb; lb++)
        { /* go through L-factor */
            ib = lb * Pc + mycol;
            index = Llu->Lrowind_bc_ptr[lb];
            if (index)
            {
                k = BC_HEADER;
                for (j = 0; j < index[0]; j++)
                {
                    jb = index[k];
                    if (jb != ib)
                        look_ahead_l[jb] =
                            SUPERLU_MAX(iperm_c_supno[ib], look_ahead_l[jb]);
                    k += LB_DESCRIPTOR + index[k + 1];
                }
            }
        }
        if (mycol < nsupers % grid->npcol)
        { /* leftover block columns */
            ib = ncb * Pc + mycol;
            index = Llu->Lrowind_bc_ptr[ncb];
            if (index)
            {
                k = BC_HEADER;
                for (j = 0; j < index[0]; j++)
                {
                    jb = index[k];
                    if (jb != ib)
                        look_ahead_l[jb] =
                            SUPERLU_MAX(iperm_c_supno[ib], look_ahead_l[jb]);
                    k += LB_DESCRIPTOR + index[k + 1];
                }
            }
        }
    }
    MPI_Allreduce(look_ahead_l, look_ahead, nsupers, MPI_INT, MPI_MAX, grid->comm);
    SUPERLU_FREE(look_ahead_l);

    iperm_u = SUPERLU_MALLOC(nsupers * sizeof(int_t));
    perm_u = SUPERLU_MALLOC(nsupers * sizeof(int_t));
    log_memory(nsupers * iword, stat);

    k = sp_ienv_dist(3, options); /* max supernode size */
    /* Instead of half storage, we'll do full storage */
    if (!(Llu->ujrow = doubleCalloc_dist(k * k)))
        ABORT("Malloc fails for ujrow[].");
    log_memory(k * k * iword, stat);

    Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    Unzval_br_ptr = Llu->Unzval_br_ptr;
    ToRecv = Llu->ToRecv;
    ToSendD = Llu->ToSendD;
    ToSendR = Llu->ToSendR;

    ldt = sp_ienv_dist(3, options); /* Size of maximum supernode */
    k = CEILING(nsupers, Pr);       /* Number of local block rows */

    /* Following code is for finding maximum row dimension of all L panels */
    int local_max_row_size = 0;
    int max_row_size;

    for (i = mycol; i < nsupers; i += Pc)
    { /* grab my local columns */
        // int tpc = PCOL (i, grid);
        lk = LBj(i, grid);
        lsub = Lrowind_bc_ptr[lk];
        if (lsub != NULL)
        {
            if (lsub[1] > local_max_row_size)
                local_max_row_size = lsub[1];
        }
    }

    /* Max row size is global reduction within a row */
    MPI_Allreduce(&local_max_row_size, &max_row_size, 1, MPI_INT, MPI_MAX,
                  (grid->rscp.comm));

    /* Buffer size is max of look-ahead window */
    /* int_t buffer_size =
         SUPERLU_MAX (max_row_size * num_threads * ldt,
                      get_max_buffer_size ());           */

    int_t max_ncols = 0;
    int_t bigu_size = estimate_bigu_size(nsupers, Ufstnz_br_ptr, Glu_persist,
                                         grid, perm_u, &max_ncols);

    /* +16 to avoid cache line false sharing */
    // int_t bigv_size = SUPERLU_MAX(max_row_size * (bigu_size / ldt),
    int_t bigv_size = SUPERLU_MAX(max_row_size * max_ncols,
                                  (ldt * ldt + CACHELINE / dword) * num_threads);

    /* bigU and bigV are only allocated on CPU, but may be allocated as
       page-locked memory accessible to GPU. */
    double *bigU; /* for storing entire U(k,:) panel, prepare for GEMM. */
    double *bigV; /* for storing GEMM output matrix, i.e. update matrix.
                  bigV is large enough to hold the aggregate GEMM output.*/
    bigU = NULL;  /* allocated only on CPU */
    bigV = NULL;

#ifdef GPU_ACC                                           /*-- use GPU --*/
    int superlu_acc_offload = sp_ienv_dist(10, options); // get_acc_offload();

    int gpublas_nb = get_gpublas_nb();    // default 64
    int nstreams = get_num_gpu_streams(); // default 8

    int_t buffer_size = SUPERLU_MIN(max_row_size * max_ncols, sp_ienv_dist(8, options));
    //   get_max_buffer_size());
    double *dA, *dB, *dC; // GEMM matrices on device
    int *stream_end_col;
    gpuError_t gpuStat;
    gpublasHandle_t *handle;
    gpuStream_t *streams;

    if (superlu_acc_offload)
    {

        /* array holding last column blk for each partition,
           used in SchCompUdt-GPU.c         */
        // int *stream_end_col = (int_t *) _mm_malloc (sizeof (int_t) * nstreams,64);
        stream_end_col = SUPERLU_MALLOC(nstreams * sizeof(int));

        if (checkGPU(gpuHostMalloc((void **)&bigU, bigu_size * sizeof(double), gpuHostMallocDefault)))
            ABORT("Malloc fails for dgemm buffer U ");

        if (checkGPU(gpuHostMalloc((void **)&bigV, bigv_size * sizeof(double), gpuHostMallocDefault)))
            ABORT("Malloc fails for dgemm buffer V");

        if (iam == 0 && options->PrintStat == YES)
        {
            DisplayHeader();
            printf(" Starting with %d GPU Streams \n", nstreams);
            fflush(stdout);
        }

        handle = (gpublasHandle_t *)SUPERLU_MALLOC(sizeof(gpublasHandle_t) * nstreams);
        for (i = 0; i < nstreams; i++)
            handle[i] = create_handle();

        // creating streams
        streams = (gpuStream_t *)SUPERLU_MALLOC(sizeof(gpuStream_t) * nstreams);
        for (i = 0; i < nstreams; ++i)
            checkGPU(gpuStreamCreate(&streams[i]));

        gpuStat = gpuMalloc((void **)&dA, max_row_size * sp_ienv_dist(3, options) * sizeof(double));
        if (gpuStat != gpuSuccess)
        {
            fprintf(stderr, "!!!! Error in allocating A in the device %ld \n", m * k * sizeof(double));
            return 1;
        }

        // size of B should be bigu_size
        gpuStat = gpuMalloc((void **)&dB, bigu_size * sizeof(double));
        if (gpuStat != gpuSuccess)
        {
            fprintf(stderr, "!!!! Error in allocating B in the device %ld \n", n * k * sizeof(double));
            return 1;
        }

        gpuStat = gpuMalloc((void **)&dC, buffer_size * sizeof(double));
        if (gpuStat != gpuSuccess)
        {
            fprintf(stderr, "!!!! Error in allocating C in the device \n");
            return 1;
        }

        stat->gpu_buffer += dword * (max_row_size * sp_ienv_dist(3, options) // dA
                                     + bigu_size                             // dB
                                     + buffer_size);                         // dC
    }
    else
    { /* now superlu_acc_offload==0, GEMM will use CPU buffer */
        if (!(bigU = doubleMalloc_dist(bigu_size)))
            ABORT("Malloc fails for dgemm U buffer");
        if (!(bigV = doubleMalloc_dist(bigv_size)))
            ABORT("Malloc failed for dgemm V buffer");
    }

#else /*-------- not to use GPU --------*/

    // for GEMM padding 0
    j = bigu_size / ldt;
    bigu_size += (gemm_k_pad * (j + ldt + gemm_n_pad));
    bigv_size += (gemm_m_pad * (j + max_row_size + gemm_n_pad));

    if (!(bigU = doubleMalloc_dist(bigu_size)))
        ABORT("Malloc fails for dgemm U buffer");
    if (!(bigV = doubleMalloc_dist(bigv_size)))
        ABORT("Malloc failed for dgemm V buffer");

#endif /*************** end ifdef GPU_ACC ****************/

    log_memory((bigv_size + bigu_size) * dword, stat);

    /* Sherry: (ldt + 16), avoid cache line false sharing.
       KNL cacheline size = 64 bytes = 16 int */
    iinfo = ldt + CACHELINE / sizeof(int);
    if (!(indirect = SUPERLU_MALLOC(iinfo * num_threads * sizeof(int))))
        ABORT("Malloc fails for indirect[].");
    if (!(indirect2 = SUPERLU_MALLOC(iinfo * num_threads * sizeof(int))))
        ABORT("Malloc fails for indirect[].");

    log_memory(2 * ldt * ldt * dword + 2 * iinfo * num_threads * iword, stat);

    int_t *lookAheadFullRow, *lookAheadStRow, *lookAhead_lptr, *lookAhead_ib,
        *RemainStRow, *Remain_lptr, *Remain_ib;

    lookAheadFullRow = intMalloc_dist((num_look_aheads + 1));
    lookAheadStRow = intMalloc_dist((num_look_aheads + 1));
    lookAhead_lptr = intMalloc_dist((num_look_aheads + 1));
    lookAhead_ib = intMalloc_dist((num_look_aheads + 1));

    int_t mrb = (nsupers + Pr - 1) / Pr;
    int_t mcb = (nsupers + Pc - 1) / Pc;

    RemainStRow = intMalloc_dist(mrb);
    Remain_lptr = intMalloc_dist(mrb);
    Remain_ib = intMalloc_dist(mrb);

    Remain_info_t *Remain_info;
    Remain_info = (Remain_info_t *)SUPERLU_MALLOC(mrb * sizeof(Remain_info_t));

    double *lookAhead_L_buff, *Remain_L_buff; /* Stores entire L-panel */
    Ublock_info_t *Ublock_info;
    ldt = sp_ienv_dist(3, options); /* max supernode size */
    /* The following is quite loose */
    lookAhead_L_buff = doubleMalloc_dist(ldt * ldt * (num_look_aheads + 1));

    j = gemm_m_pad * (ldt + max_row_size + gemm_k_pad);
    Remain_L_buff = doubleMalloc_dist(Llu->bufmax[1] + j); /* This is loose */
    Ublock_info = (Ublock_info_t *)SUPERLU_MALLOC(mcb * sizeof(Ublock_info_t));

    long long alloc_mem = 3 * mrb * iword + mrb * sizeof(Remain_info_t) + ldt * ldt * (num_look_aheads + 1) * dword + Llu->bufmax[1] * dword;
    log_memory(alloc_mem, stat);

    InitTimer = SuperLU_timer_() - tt1;

    double pxgstrfTimer = SuperLU_timer_();

    /* ##################################################################
       ** Handle first block column separately to start the pipeline. **
       ################################################################## */
    look_id = 0;
    msgcnt = msgcnts[0]; /* Lsub[0] to be transferred */
    send_req = send_reqs[0];
    recv_req = recv_reqs[0];

    k0 = 0;
    k = perm_c_supno[0];
    kcol = PCOL(k, grid);
    krow = PROW(k, grid);
    if (mycol == kcol)
    {
        double ttt1 = SuperLU_timer_();
        /* panel factorization */
        PDGSTRF2(options, k0, k, thresh, Glu_persist, grid, Llu,
                 U_diag_blk_send_req, tag_ub, stat, info);

        pdgstrf2_timer += SuperLU_timer_() - ttt1;

        scp = &grid->rscp; /* The scope of process row. */

        /* Multicasts numeric values of L(:,0) to process rows. */
        lk = LBj(k, grid); /* Local block number. */ // (Yida) : lk is pbcol, <= ceil(block_length / q)
        lsub = Lrowind_bc_ptr[lk];
        lusup = Lnzval_bc_ptr[lk];
        if (lsub)
        {
            /* number of entries in Lsub_buf[] to be transferred */
            msgcnt[0] = lsub[1] + BC_HEADER + lsub[0] * LB_DESCRIPTOR;
            /* number of entries in Lval_buf[] to be transferred */
            msgcnt[1] = lsub[1] * SuperSize(k);

            printf("glo_supcol = %d, loc_supcol = %d, nzblk_supcol = %d, sup_width = %d, SuperSize(k) = %d\n", 
                k, lk, lsub[0], lsub[1], SuperSize(k));
        }
        else
        {
            msgcnt[0] = msgcnt[1] = 0;
        }

        for (pj = 0; pj < Pc; ++pj)
        {
            if (ToSendR[lk][pj] != SLU_EMPTY) // (Yida) : ToSendR 记录了每个pbcol，每个进程是否需要对角元
            {
                int rank = -1;
                MPI_Comm_rank(scp->comm, &rank);
                // printf("Send %d->%d\n", rank, pj);
                MPI_Isend(lsub, msgcnt[0], mpi_int_t, pj,
                          SLU_MPI_TAG(0, 0) /* 0 */,
                          scp->comm, &send_req[pj]);
                MPI_Isend(lusup, msgcnt[1], MPI_DOUBLE, pj,
                          SLU_MPI_TAG(1, 0) /* 1 */,
                          scp->comm, &send_req[pj + Pc]);
            } /* end if */
        } /* end for pj ... */
    }
    else
    { /* Post immediate receives. */
        if (ToRecv[k] >= 1)
        {                      /* Recv block column L(:,0). */
            scp = &grid->rscp; /* The scope of process row. */
            MPI_Irecv(Lsub_buf_2[0], Llu->bufmax[0], mpi_int_t, kcol,
                      SLU_MPI_TAG(0, 0) /* 0 */,
                      scp->comm, &recv_req[0]);
            MPI_Irecv(Lval_buf_2[0], Llu->bufmax[1], MPI_DOUBLE, kcol,
                      SLU_MPI_TAG(1, 0) /* 1 */,
                      scp->comm, &recv_req[1]);
        }
    } /* end if mycol == 0 */

    factored[k] = 0; /* flag column k as factored. */

    /* post receive of first U-row */ // (Yida) : ?
    if (myrow != krow)
    {
        if (ToRecv[k] == 2)
        {                      /* Recv block row U(k,:). */
            scp = &grid->cscp; /* The scope of process column. */
            Usub_buf = Llu->Usub_buf_2[0];
            Uval_buf = Llu->Uval_buf_2[0];
            MPI_Irecv(Usub_buf, Llu->bufmax[2], mpi_int_t, krow,
                      SLU_MPI_TAG(2, 0) /* 2%tag_ub */,
                      scp->comm, &recv_reqs_u[0][0]);
            MPI_Irecv(Uval_buf, Llu->bufmax[3], MPI_DOUBLE, krow,
                      SLU_MPI_TAG(3, 0) /* 3%tag_ub */,
                      scp->comm, &recv_reqs_u[0][1]);
        }
    }

    /* ##################################################################
       **** MAIN LOOP ****
       ################################################################## */
    for (k0 = 0; k0 < nsupers; ++k0)
    {
        k = perm_c_supno[k0];

        /* ============================================ *
         * ======= look-ahead the new L columns ======= *
         * ============================================ */
        if (k0 == 0)
        { /* look-ahead all the columns in the window */
            kk1 = k0 + 1;
            kk2 = SUPERLU_MIN(k0 + num_look_aheads, nsupers - 1);
        }
        else
        { /* look-ahead one new column after the current window */
            kk1 = k0 + num_look_aheads;
            kk2 = SUPERLU_MIN(kk1, nsupers - 1);
        }

        for (kk0 = kk1; kk0 <= kk2; kk0++)
        {
            /* loop through look-ahead window in L */

            kk = perm_c_supno[kk0];                /* use the ordering from static schedule */
            look_id = kk0 % (1 + num_look_aheads); /* which column in window */

            if (look_ahead[kk] < k0)
            { /* does not depend on current column k */
                kcol = PCOL(kk, grid);
                if (mycol == kcol)
                { /* I own this panel */

                    /* Panel factorization -- Factor diagonal and subdiagonal
                       L blocks and test for exact singularity.  */
                    factored[kk] = 0; /* flag column kk as factored */
                    double ttt1 = SuperLU_timer_();

                    PDGSTRF2(options, kk0, kk, thresh, Glu_persist,
                             grid, Llu, U_diag_blk_send_req, tag_ub, stat, info); // (Yida) : GETRF + TSTRF

                    pdgstrf2_timer += SuperLU_timer_() - ttt1;

                    /* Multicasts numeric values of L(:,kk) to process rows. */
                    /* ttt1 = SuperLU_timer_(); */
                    msgcnt = msgcnts[look_id]; /* point to the proper count array */
                    send_req = send_reqs[look_id];

                    lk = LBj(kk, grid); /* Local block number in L. */
                    lsub1 = Lrowind_bc_ptr[lk];
                    if (lsub1)
                    {
                        msgcnt[0] = lsub1[1] + BC_HEADER + lsub1[0] * LB_DESCRIPTOR; /* size of metadata */
                        msgcnt[1] = lsub1[1] * SuperSize(kk);                        /* Lval_buf[] size */
                    }
                    else
                    {
                        msgcnt[0] = 0;
                        msgcnt[1] = 0;
                    }
                    scp = &grid->rscp; /* The scope of process row. */
                    for (pj = 0; pj < Pc; ++pj)
                    {
                        if (ToSendR[lk][pj] != SLU_EMPTY)
                        {
                            lusup1 = Lnzval_bc_ptr[lk];
                            MPI_Isend(lsub1, msgcnt[0], mpi_int_t, pj,
                                      SLU_MPI_TAG(0, kk0), /* (4*kk0)%tag_ub */
                                      scp->comm, &send_req[pj]);
                            MPI_Isend(lusup1, msgcnt[1], MPI_DOUBLE, pj,
                                      SLU_MPI_TAG(1, kk0), /* (4*kk0+1)%tag_ub */
                                      scp->comm, &send_req[pj + Pc]);
                        }
                    }
                    /* stat->time9 += SuperLU_timer_() - ttt1; */
                }
                else
                { /* Post Recv of block column L(:,kk). */
                    /* double ttt1 = SuperLU_timer_(); */
                    if (ToRecv[kk] >= 1)
                    {
                        scp = &grid->rscp; /* The scope of process row. */
                        recv_req = recv_reqs[look_id];
                        MPI_Irecv(Lsub_buf_2[look_id], Llu->bufmax[0],
                                  mpi_int_t, kcol, SLU_MPI_TAG(0, kk0), /* (4*kk0)%tag_ub */
                                  scp->comm, &recv_req[0]);
                        MPI_Irecv(Lval_buf_2[look_id], Llu->bufmax[1],
                                  MPI_DOUBLE, kcol,
                                  SLU_MPI_TAG(1, kk0), /* (4*kk0+1)%tag_ub */
                                  scp->comm, &recv_req[1]);
                    }
                    /* stat->time10 += SuperLU_timer_() - ttt1; */
                } /* end if mycol == Pc(kk) */
            } /* end if look-ahead in L panels */

            /* Pre-post irecv for U-row look-ahead */
            krow = PROW(kk, grid);
            if (myrow != krow)
            {
                if (ToRecv[kk] == 2)
                {                      /* post iRecv block row U(kk,:). */
                    scp = &grid->cscp; /* The scope of process column. */
                    Usub_buf = Llu->Usub_buf_2[look_id];
                    Uval_buf = Llu->Uval_buf_2[look_id];
                    MPI_Irecv(Usub_buf, Llu->bufmax[2], mpi_int_t, krow,
                              SLU_MPI_TAG(2, kk0) /* (4*kk0+2)%tag_ub */,
                              scp->comm, &recv_reqs_u[look_id][0]);
                    MPI_Irecv(Uval_buf, Llu->bufmax[3], MPI_DOUBLE, krow,
                              SLU_MPI_TAG(3, kk0) /* (4*kk0+3)%tag_ub */,
                              scp->comm, &recv_reqs_u[look_id][1]);
                }
            }

        } /* end for each column in look-ahead window for L panels */

        /* stat->time4 += SuperLU_timer_()-tt1; */

        /* ================================= *
         * ==== look-ahead the U rows    === *
         * ================================= */
        kk1 = k0;
        kk2 = SUPERLU_MIN(k0 + num_look_aheads, nsupers - 1);
        for (kk0 = kk1; kk0 < kk2; kk0++)
        {
            kk = perm_c_supno[kk0]; /* order determined from static schedule */
            if (factoredU[kk0] != 1 && look_ahead[kk] < k0)
            {
                /* does not depend on current column k */
                kcol = PCOL(kk, grid);
                krow = PROW(kk, grid);
                lk = LBj(kk, grid); /* Local block number across row. NOT USED?? -- Sherry */

                look_id = kk0 % (1 + num_look_aheads);
                msgcnt = msgcntsU[look_id];
                recv_req = recv_reqs[look_id];

                /* ================================================= *
                 * Check if diagonal block has been received         *
                 * for panel factorization of U in look-ahead window *
                 * ================================================= */

                if (mycol == kcol)
                { /* I own this column panel, no need
                     to receive L  */
                    flag0 = flag1 = 1;
                    msgcnt[0] = msgcnt[1] = -1; /* No need to transfer Lsub, nor Lval */
                }
                else
                { /* Check to receive L(:,kk) from the left */
                    flag0 = flag1 = 0;
                    if (ToRecv[kk] >= 1)
                    {
                        if (recv_req[0] != MPI_REQUEST_NULL)
                        {
                            MPI_Test(&recv_req[0], &flag0, &status);
                            if (flag0)
                            {
                                MPI_Get_count(&status, mpi_int_t, &msgcnt[0]);
                                recv_req[0] = MPI_REQUEST_NULL;
                            }
                        }
                        else
                            flag0 = 1;

                        if (recv_req[1] != MPI_REQUEST_NULL)
                        {
                            MPI_Test(&recv_req[1], &flag1, &status);
                            if (flag1)
                            {
                                MPI_Get_count(&status, mpi_int_t, &msgcnt[1]);
                                recv_req[1] = MPI_REQUEST_NULL;
                            }
                        }
                        else
                            flag1 = 1;
                    }
                    else
                    {
                        msgcnt[0] = 0;
                    }
                }

                if (flag0 && flag1)
                { /* L(:,kk) is ready */
                    /* tt1 = SuperLU_timer_(); */
                    scp = &grid->cscp; /* The scope of process column. */
                    if (myrow == krow)
                    {
                        factoredU[kk0] = 1;
                        /* Parallel triangular solve across process row *krow* --
                           U(k,j) = L(k,k) \ A(k,j).  */
                        double ttt2 = SuperLU_timer_();
                        pdgstrs2_omp(kk0, kk, Glu_persist, grid, Llu,
                                     Ublock_info, stat);
                        pdgstrs2_timer += SuperLU_timer_() - ttt2;

                        /* Multicasts U(kk,:) to process columns. */
                        lk = LBi(kk, grid);
                        usub = Ufstnz_br_ptr[lk];
                        uval = Unzval_br_ptr[lk];
                        if (usub)
                        {
                            msgcnt[2] = usub[2]; /* metadata size */
                            msgcnt[3] = usub[1]; /* Uval[] size */
                        }
                        else
                        {
                            msgcnt[2] = msgcnt[3] = 0;
                        }

                        if (ToSendD[lk] == YES)
                        {
                            for (pi = 0; pi < Pr; ++pi)
                            {
                                if (pi != myrow)
                                {
                                    MPI_Isend(usub, msgcnt[2], mpi_int_t, pi,
                                              SLU_MPI_TAG(2, kk0), /* (4*kk0+2)%tag_ub */
                                              scp->comm, &send_reqs_u[look_id][pi]);
                                    MPI_Isend(uval, msgcnt[3], MPI_DOUBLE,
                                              pi, SLU_MPI_TAG(3, kk0), /* (4*kk0+3)%tag_ub */
                                              scp->comm, &send_reqs_u[look_id][pi + Pr]);
                                } /* if pi ... */
                            } /* for pi ... */
                        } /* if ToSendD ... */

                        /* stat->time2 += SuperLU_timer_()-tt1; */

                    } /* end if myrow == krow */
                } /* end if flag0 & flag1 ... */
            } /* end if factoredU[] ... */
        } /* end for kk0 ... */

        /* ============================================== *
         * == start processing the current row of U(k,:) *
         * ============================================== */
        knsupc = SuperSize(k);
        krow = PROW(k, grid);
        kcol = PCOL(k, grid);

        /* tt1 = SuperLU_timer_(); */
        look_id = k0 % (1 + num_look_aheads);
        recv_req = recv_reqs[look_id];
        send_req = send_reqs[look_id];
        msgcnt = msgcnts[look_id];
        Usub_buf = Llu->Usub_buf_2[look_id];
        Uval_buf = Llu->Uval_buf_2[look_id];

        if (mycol == kcol)
        {
            lk = LBj(k, grid); /* Local block number in L */
            for (pj = 0; pj < Pc; ++pj)
            {
                /* Wait for Isend to complete before using lsub/lusup buffer. */
                if (ToSendR[lk][pj] != SLU_EMPTY)
                {
                    MPI_Wait(&send_req[pj], &status);
                    MPI_Wait(&send_req[pj + Pc], &status);
                }
            }
            lsub = Lrowind_bc_ptr[lk];
            lusup = Lnzval_bc_ptr[lk];
        }
        else
        {
            if (ToRecv[k] >= 1)
            { /* Recv block column L(:,k). */

                scp = &grid->rscp; /* The scope of process row. */

                /* ============================================= *
                 * Waiting for L(:,kk) for outer-product uptate  *
                 * if iam in U(kk,:), then the diagonal block    *
                 * did not reach in time for panel factorization *
                 * of U(k,:).          	                         *
                 * ============================================= */
                if (recv_req[0] != MPI_REQUEST_NULL)
                {
                    MPI_Wait(&recv_req[0], &status);
                    MPI_Get_count(&status, mpi_int_t, &msgcnt[0]);
                    recv_req[0] = MPI_REQUEST_NULL;
                }
                else
                {
                    msgcnt[0] = msgcntsU[look_id][0];
                }

                if (recv_req[1] != MPI_REQUEST_NULL)
                {
                    MPI_Wait(&recv_req[1], &status);
                    MPI_Get_count(&status, MPI_DOUBLE, &msgcnt[1]);
                    recv_req[1] = MPI_REQUEST_NULL;
                }
                else
                {
                    msgcnt[1] = msgcntsU[look_id][1];
                }
            }
            else
            {
                msgcnt[0] = 0;
            }

            lsub = Lsub_buf_2[look_id];
            lusup = Lval_buf_2[look_id];
        } /* else if mycol = Pc(k) */
        /* stat->time1 += SuperLU_timer_()-tt1; */

        scp = &grid->cscp; /* The scope of process column. */

        /* tt1 = SuperLU_timer_(); */
        if (myrow == krow)
        { /* I own U(k,:) */
            lk = LBi(k, grid);
            usub = Ufstnz_br_ptr[lk];
            uval = Unzval_br_ptr[lk];

            if (factoredU[k0] == -1)
            {
                /* Parallel triangular solve across process row *krow* --
                   U(k,j) = L(k,k) \ A(k,j).  */
                // (Yida) : GESSM
                double ttt2 = SuperLU_timer_();
                pdgstrs2_omp(k0, k, Glu_persist, grid, Llu, Ublock_info, stat);
                pdgstrs2_timer += SuperLU_timer_() - ttt2;

                /* Sherry -- need to set factoredU[k0] = 1; ?? */

                /* Multicasts U(k,:) along process columns. */
                if (usub)
                {
                    msgcnt[2] = usub[2]; /* metadata size */
                    msgcnt[3] = usub[1]; /* Uval[] size */
                }
                else
                {
                    msgcnt[2] = msgcnt[3] = 0;
                }

                if (ToSendD[lk] == YES)
                {
                    for (pi = 0; pi < Pr; ++pi)
                    {
                        if (pi != myrow)
                        { /* Matching recv was pre-posted before */
                            MPI_Send(usub, msgcnt[2], mpi_int_t, pi,
                                     SLU_MPI_TAG(2, k0), /* (4*k0+2)%tag_ub */
                                     scp->comm);
                            MPI_Send(uval, msgcnt[3], MPI_DOUBLE, pi,
                                     SLU_MPI_TAG(3, k0), /* (4*k0+3)%tag_ub */
                                     scp->comm);
                        } /* if pi ... */
                    } /* for pi ... */
                } /* if ToSendD ... */
            }
            else
            { /* Panel U(k,:) already factorized from previous look-ahead */

                /* ================================================ *
                 * Wait for downward sending of U(k,:) to complete  *
                 * for outer-product update.                        *
                 * ================================================ */

                if (ToSendD[lk] == YES)
                {
                    for (pi = 0; pi < Pr; ++pi)
                    {
                        if (pi != myrow)
                        {
                            MPI_Wait(&send_reqs_u[look_id][pi], &status);
                            MPI_Wait(&send_reqs_u[look_id][pi + Pr], &status);
                        }
                    }
                }
                msgcnt[2] = msgcntsU[look_id][2];
                msgcnt[3] = msgcntsU[look_id][3];
            }
            /* stat->time2 += SuperLU_timer_()-tt1; */
        }
        else
        { /* myrow != krow */

            /* ========================================== *
             * Wait for U(k,:) for outer-product updates. *
             * ========================================== */

            if (ToRecv[k] == 2)
            { /* Recv block row U(k,:). */
                MPI_Wait(&recv_reqs_u[look_id][0], &status);
                MPI_Get_count(&status, mpi_int_t, &msgcnt[2]);
                MPI_Wait(&recv_reqs_u[look_id][1], &status);
                MPI_Get_count(&status, MPI_DOUBLE, &msgcnt[3]);
                usub = Usub_buf;
                uval = Uval_buf;
            }
            else
            {
                msgcnt[2] = 0;
            }
            /* stat->time6 += SuperLU_timer_()-tt1; */
        } /* end if myrow == Pr(k) */

        /*
         * Parallel rank-k update; pair up blocks L(i,k) and U(k,j).
         *  for (j = k+1; k < N; ++k) {
         *     for (i = k+1; i < N; ++i)
         *         if ( myrow == PROW( i, grid ) && mycol == PCOL( j, grid )
         *              && L(i,k) != 0 && U(k,j) != 0 )
         *             A(i,j) = A(i,j) - L(i,k) * U(k,j);
         */
        msg0 = msgcnt[0];
        msg2 = msgcnt[2];
        /* tt1 = SuperLU_timer_(); */
        if (msg0 && msg2)
        {                    /* L(:,k) and U(k,:) are not empty. */
            nsupr = lsub[1]; /* LDA of lusup. */
            if (myrow == krow)
            { /* Skip diagonal block L(k,k). */
                lptr0 = BC_HEADER + LB_DESCRIPTOR + lsub[BC_HEADER + 1];
                luptr0 = knsupc;
                nlb = lsub[0] - 1;
            }
            else
            {
                lptr0 = BC_HEADER;
                luptr0 = 0;
                nlb = lsub[0];
            }
            iukp = BR_HEADER; /* Skip header; Pointer to index[] of U(k,:) */
            rukp = 0;         /* Pointer to nzval[] of U(k,:) */
            nub = usub[0];    /* Number of blocks in the block row U(k,:) */
            klst = FstBlockC(k + 1);

            /* -------------------------------------------------------------
               Update the look-ahead block columns A(:,k+1:k+num_look_ahead)
               ------------------------------------------------------------- */
            iukp0 = iukp;
            rukp0 = rukp;
            /* reorder the remaining columns in bottome-up */
            /* TAU_STATIC_TIMER_START("LOOK_AHEAD_UPDATE"); */
            for (jj = 0; jj < nub; jj++)
            {
                iperm_u[jj] = iperm_c_supno[usub[iukp]]; /* Global block number of block U(k,j). */
                perm_u[jj] = jj;
                jb = usub[iukp]; /* Global block number of block U(k,j). */
                nsupc = SuperSize(jb);
                iukp += UB_DESCRIPTOR; /* Start fstnz of block U(k,j). */
                iukp += nsupc;
            }
            iukp = iukp0;
            /* iperm_u is sorted based on elimination order;
               perm_u reorders the U blocks to match the elimination order. */
            isort(nub, iperm_u, perm_u);

            /************************************************************************/
            double ttx = SuperLU_timer_();

// #include "dlook_ahead_update.c"
#include <assert.h> /* assertion doesn't work if NDEBUG is defined */

            iukp = iukp0; /* point to the first block in index[] */
            rukp = rukp0; /* point to the start of nzval[] */
            j = jj0 = 0;  /* After the j-loop, jj0 points to the first block in U
                             outside look-ahead window. */

#ifdef ISORT
            while (j < nub && iperm_u[j] <= k0 + num_look_aheads)
#else
            while (j < nub && perm_u[2 * j] <= k0 + num_look_aheads)
#endif
            {
                double zero = 0.0;
                /* Search is needed because a permutation perm_u is involved for j  */
                /* Search along the row for the pointers {iukp, rukp} pointing to
                 * block U(k,j).
                 * j    -- current block in look-ahead window, initialized to 0 on entry
                 * iukp -- point to the start of index[] metadata
                 * rukp -- point to the start of nzval[] array
                 * jb   -- block number of block U(k,j), update destination column
                 */
                arrive_at_ublock(
                    j, &iukp, &rukp, &jb, &ljb, &nsupc,
                    iukp0, rukp0, usub, perm_u, xsup, grid);

                j++;
                jj0++;
                jj = iukp;

                while (usub[jj] == klst)
                    ++jj; /* Skip zero segments */

                ldu = klst - usub[jj++];
                ncols = 1;

                /* This loop computes ldu. */
                for (; jj < iukp + nsupc; ++jj)
                { /* for each column jj in block U(k,j) */
                    segsize = klst - usub[jj];
                    if (segsize)
                    {
                        ++ncols;
                        if (segsize > ldu)
                            ldu = segsize;
                    }
                }

                /* Now copy one block U(k,j) to bigU for GEMM, padding zeros up to ldu. */
                tempu = bigU; /* Copy one block U(k,j) to bigU for GEMM */
                for (jj = iukp; jj < iukp + nsupc; ++jj)
                {
                    segsize = klst - usub[jj];
                    if (segsize)
                    {
                        lead_zero = ldu - segsize;
                        for (i = 0; i < lead_zero; ++i)
                            tempu[i] = zero;
                        tempu += lead_zero;
                        for (i = 0; i < segsize; ++i)
                        {
                            tempu[i] = uval[rukp + i];
                        }
                        rukp += segsize;
                        tempu += segsize;
                    }
                }
                tempu = bigU; /* set back to the beginning of the buffer */

                nbrow = lsub[1]; /* number of row subscripts in L(:,k) */
                if (myrow == krow)
                    nbrow = lsub[1] - lsub[3]; /* skip diagonal block for those rows. */
                // double ttx =SuperLU_timer_();

                int current_b = 0; /* Each thread starts searching from first block.
                                      This records the moving search target.           */
                lptr = lptr0;      /* point to the start of index[] in supernode L(:,k) */
                luptr = luptr0;

/* Sherry -- examine all the shared variables ??
'firstprivate' ensures that the private variables are initialized
to the values before entering the loop.  */
#pragma omp parallel for firstprivate(lptr, luptr, current_b) private(ib, lb) default(shared) schedule(dynamic)
                for (lb = 0; lb < nlb; lb++)
                {                   /* Loop through each block in L(:,k) */
                    int temp_nbrow; /* automatic variable is private */

                    /* Search for the L block that my thread will work on.
                       No need to search from 0, can continue at the point where
                       it is left from last iteration.
                       Note: Blocks may not be sorted in L. Different thread picks up
                       different lb.   */
                    for (; current_b < lb; ++current_b)
                    {
                        temp_nbrow = lsub[lptr + 1]; /* Number of full rows. */
                        lptr += LB_DESCRIPTOR;       /* Skip descriptor. */
                        lptr += temp_nbrow;          /* move to next block */
                        luptr += temp_nbrow;         /* move to next block */
                    }

                    int_t thread_id = omp_get_thread_num();
                    double *tempv = bigV + ldt * ldt * thread_id;

                    int *indirect_thread = indirect + ldt * thread_id;
                    int *indirect2_thread = indirect2 + ldt * thread_id;
                    ib = lsub[lptr];             /* block number of L(i,k) */
                    temp_nbrow = lsub[lptr + 1]; /* Number of full rows. */
                                                 /* assert (temp_nbrow <= nbrow); */

                    lptr += LB_DESCRIPTOR; /* Skip descriptor. */

                    /*if (thread_id == 0) tt_start = SuperLU_timer_();*/

                    /* calling gemm */
                    stat->ops[FACT] += 2.0 * (flops_t)temp_nbrow * ldu * ncols;
                    dgemm_(
                        "N", "N", 
                        &temp_nbrow, &ncols, &ldu, // (Yida) : m, n, k
                        &alpha, 
                        &lusup[luptr + (knsupc - ldu) * nsupr], &nsupr, // (Yida) : a, lda
                        tempu, &ldu, // (Yida) : b, ldb
                        &beta, 
                        tempv, &temp_nbrow, // (Yida) : c, ldc
                        1, 1
                    );

                    /* Now scattering the output. */
                    if (ib < jb)
                    { /* A(i,j) is in U. */
                        dscatter_u(ib, jb,
                                   nsupc, iukp, xsup,
                                   klst, temp_nbrow,
                                   lptr, temp_nbrow, lsub,
                                   usub, tempv, Ufstnz_br_ptr, Unzval_br_ptr, grid);
                    }
                    else
                    { /* A(i,j) is in L. */
                        dscatter_l(ib, ljb, nsupc, iukp, xsup, klst, temp_nbrow, lptr,
                                   temp_nbrow, usub, lsub, tempv,
                                   indirect_thread, indirect2_thread,
                                   Lrowind_bc_ptr, Lnzval_bc_ptr, grid);
                    }

                    ++current_b; /* Move to next block. */
                    lptr += temp_nbrow;
                    luptr += temp_nbrow;
                } /* end parallel for lb = 0, nlb ... all blocks in L(:,k) */

                iukp += nsupc; /* Mov to block U(k,j+1) */

                /* =========================================== *
                 * == factorize L(:,j) and send if possible == *
                 * =========================================== */
                kk = jb; /* destination column that is just updated */
                kcol = PCOL(kk, grid);
#ifdef ISORT
                kk0 = iperm_u[j - 1];
#else
                kk0 = perm_u[2 * (j - 1)];
#endif
                look_id = kk0 % (1 + num_look_aheads);

                if (look_ahead[kk] == k0 && kcol == mycol)
                {
                    /* current column is the last dependency */
                    look_id = kk0 % (1 + num_look_aheads);

                    /* Factor diagonal and subdiagonal blocks and test for exact
                       singularity.  */
                    factored[kk] = 0;

                    double tt1 = SuperLU_timer_();

                    PDGSTRF2(options, kk0, kk, thresh, Glu_persist, grid, Llu,
                             U_diag_blk_send_req, tag_ub, stat, info);

                    pdgstrf2_timer += SuperLU_timer_() - tt1;

                    /* stat->time7 += SuperLU_timer_() - ttt1; */

                    /* Multicasts numeric values of L(:,kk) to process rows. */
                    send_req = send_reqs[look_id];
                    msgcnt = msgcnts[look_id];

                    lk = LBj(kk, grid); /* Local block number. */
                    lsub1 = Lrowind_bc_ptr[lk];
                    lusup1 = Lnzval_bc_ptr[lk];
                    if (lsub1)
                    {
                        msgcnt[0] = lsub1[1] + BC_HEADER + lsub1[0] * LB_DESCRIPTOR;
                        msgcnt[1] = lsub1[1] * SuperSize(kk);
                    }
                    else
                    {
                        msgcnt[0] = 0;
                        msgcnt[1] = 0;
                    }

                    scp = &grid->rscp; /* The scope of process row. */
                    for (pj = 0; pj < Pc; ++pj)
                    {
                        if (ToSendR[lk][pj] != SLU_EMPTY)
                        {
                            MPI_Isend(lsub1, msgcnt[0], mpi_int_t, pj,
                                      SLU_MPI_TAG(0, kk0) /* (4*kk0)%tag_ub */,
                                      scp->comm, &send_req[pj]);
                            MPI_Isend(lusup1, msgcnt[1], MPI_DOUBLE, pj,
                                      SLU_MPI_TAG(1, kk0) /* (4*kk0+1)%tag_ub */,
                                      scp->comm, &send_req[pj + Pc]);
                        } /* end if ( ToSendR[lk][pj] != SLU_EMPTY ) */
                    } /* end for pj ... */
                } /* end if( look_ahead[kk] == k0 && kcol == mycol ) */
            } /* end while j < nub and perm_u[j] <k0+NUM_LOOK_AHEAD */
            // end #include "dlook_ahead_update.c"

            lookaheadupdatetimer += SuperLU_timer_() - ttx;
            /************************************************************************/

            /*ifdef OMP_LOOK_AHEAD */
            /* TAU_STATIC_TIMER_STOP("LOOK_AHEAD_UPDATE"); */
        } /* if L(:,k) and U(k,:) not empty */

        /* stat->time3 += SuperLU_timer_()-tt1; */

        /* ================== */
        /* == post receive == */
        /* ================== */
        kk1 = SUPERLU_MIN(k0 + num_look_aheads, nsupers - 1);
        for (kk0 = k0 + 1; kk0 <= kk1; kk0++)
        {
            kk = perm_c_supno[kk0];
            kcol = PCOL(kk, grid);

            if (look_ahead[kk] == k0)
            {
                if (mycol != kcol)
                {
                    if (ToRecv[kk] >= 1)
                    {
                        scp = &grid->rscp; /* The scope of process row. */

                        look_id = kk0 % (1 + num_look_aheads);
                        recv_req = recv_reqs[look_id];
                        MPI_Irecv(Lsub_buf_2[look_id], Llu->bufmax[0],
                                  mpi_int_t, kcol, SLU_MPI_TAG(0, kk0), /* (4*kk0)%tag_ub */
                                  scp->comm, &recv_req[0]);
                        MPI_Irecv(Lval_buf_2[look_id], Llu->bufmax[1],
                                  MPI_DOUBLE, kcol,
                                  SLU_MPI_TAG(1, kk0), /* (4*kk0+1)%tag_ub */
                                  scp->comm, &recv_req[1]);
                    }
                }
                else
                {
                    lk = LBj(kk, grid); /* Local block number. */
                    lsub1 = Lrowind_bc_ptr[lk];
                    lusup1 = Lnzval_bc_ptr[lk];
                    if (factored[kk] == -1)
                    {
                        /* Factor diagonal and subdiagonal blocks and
               test for exact singularity.  */
                        factored[kk] = 0; /* flag column kk as factored */
                        double ttt1 = SuperLU_timer_();
                        PDGSTRF2(options, kk0, kk, thresh,
                                 Glu_persist, grid, Llu, U_diag_blk_send_req,
                                 tag_ub, stat, info);
                        pdgstrf2_timer += SuperLU_timer_() - ttt1;

                        /* Process column *kcol+1* multicasts numeric
               values of L(:,k+1) to process rows. */
                        look_id = kk0 % (1 + num_look_aheads);
                        send_req = send_reqs[look_id];
                        msgcnt = msgcnts[look_id];

                        if (lsub1)
                        {
                            msgcnt[0] = lsub1[1] + BC_HEADER + lsub1[0] * LB_DESCRIPTOR;
                            msgcnt[1] = lsub1[1] * SuperSize(kk);
                        }
                        else
                        {
                            msgcnt[0] = 0;
                            msgcnt[1] = 0;
                        }

                        scp = &grid->rscp; /* The scope of process row. */
                        for (pj = 0; pj < Pc; ++pj)
                        {
                            if (ToSendR[lk][pj] != SLU_EMPTY)
                            {
                                MPI_Isend(lsub1, msgcnt[0], mpi_int_t, pj,
                                          SLU_MPI_TAG(0, kk0), /* (4*kk0)%tag_ub */
                                          scp->comm, &send_req[pj]);
                                MPI_Isend(lusup1, msgcnt[1], MPI_DOUBLE, pj,
                                          SLU_MPI_TAG(1, kk0), /* (4*kk0+1)%tag_ub */
                                          scp->comm, &send_req[pj + Pc]);
                            }
                        } /* end for pj ... */
                    } /* if    factored[kk] ... */
                }
            }
        }

        double tsch = SuperLU_timer_();

        /*******************************************************************/

#ifdef GPU_ACC /*-- use GPU --*/
        if (superlu_acc_offload)
        {
#include "dSchCompUdt-gpu.c"
        }
        else
        {
#include "dSchCompUdt-2Ddynamic.c" // This code has better OpenMP support
        }
#else
#include "dSchCompUdt-2Ddynamic.c"
#endif
        NetSchurUpTimer += SuperLU_timer_() - tsch;
    } /* MAIN LOOP for k0 = 0, ... */

    /* ##################################################################
       ** END MAIN LOOP: for k0 = ...
       ################################################################## */

    pxgstrfTimer = SuperLU_timer_() - pxgstrfTimer;

    /********************************************************
     * Free memory                                          *
     ********************************************************/

    if (Pr * Pc > 1)
    {
        SUPERLU_FREE(Lsub_buf_2[0]); /* also free Lsub_buf_2[1] */
        SUPERLU_FREE(Lval_buf_2[0]); /* also free Lval_buf_2[1] */
        if (Llu->bufmax[2] != 0)
            SUPERLU_FREE(Usub_buf_2[0]);
        if (Llu->bufmax[3] != 0)
            SUPERLU_FREE(Uval_buf_2[0]);
        if (U_diag_blk_send_req[myrow] != MPI_REQUEST_NULL)
        {
            /* wait for last Isend requests to complete, deallocate objects */
            for (krow = 0; krow < Pr; ++krow)
            {
                if (krow != myrow)
                    MPI_Wait(U_diag_blk_send_req + krow, &status);
            }
        }
        SUPERLU_FREE(U_diag_blk_send_req);
    }

    log_memory(-((Llu->bufmax[0] + Llu->bufmax[2]) * (num_look_aheads + 1) * iword +
                 (Llu->bufmax[1] + Llu->bufmax[3]) * (num_look_aheads + 1) * dword),
               stat);

    SUPERLU_FREE(Lsub_buf_2);
    SUPERLU_FREE(Lval_buf_2);
    SUPERLU_FREE(Usub_buf_2);
    SUPERLU_FREE(Uval_buf_2);
    SUPERLU_FREE(perm_c_supno);
    SUPERLU_FREE(perm_u);
    SUPERLU_FREE(iperm_u);
    SUPERLU_FREE(look_ahead);
    SUPERLU_FREE(factoredU);
    SUPERLU_FREE(factored);
    log_memory(-(6 * nsupers * iword), stat);

    for (i = 0; i <= num_look_aheads; i++)
    {
        SUPERLU_FREE(msgcnts[i]);
        SUPERLU_FREE(msgcntsU[i]);
    }
    SUPERLU_FREE(msgcnts);
    SUPERLU_FREE(msgcntsU);

    for (i = 0; i <= num_look_aheads; i++)
    {
        SUPERLU_FREE(send_reqs_u[i]);
        SUPERLU_FREE(recv_reqs_u[i]);
        SUPERLU_FREE(send_reqs[i]);
        SUPERLU_FREE(recv_reqs[i]);
    }

    SUPERLU_FREE(recv_reqs_u);
    SUPERLU_FREE(send_reqs_u);
    SUPERLU_FREE(recv_reqs);
    SUPERLU_FREE(send_reqs);

#ifdef GPU_ACC
    if (superlu_acc_offload)
    {
        checkGPU(gpuFreeHost(bigV));
        checkGPU(gpuFreeHost(bigU));
        gpuFree((void *)dA); /* Sherry added */
        gpuFree((void *)dB);
        gpuFree((void *)dC);
        for (i = 0; i < nstreams; i++)
            destroy_handle(handle[i]);
        SUPERLU_FREE(handle);
        SUPERLU_FREE(streams);
        SUPERLU_FREE(stream_end_col);
    }
    else
    {
        SUPERLU_FREE(bigV); // allocated on CPU
        SUPERLU_FREE(bigU);
    }
#else

    SUPERLU_FREE(bigV);
    SUPERLU_FREE(bigU);

    /* Decrement freed memory from memory stat. */
    log_memory(-(bigv_size + bigu_size) * dword, stat);
#endif

    SUPERLU_FREE(Llu->ujrow);
    // SUPERLU_FREE (tempv2d);/* Sherry */
    SUPERLU_FREE(indirect);
    SUPERLU_FREE(indirect2); /* Sherry added */

    ldt = sp_ienv_dist(3, options);
    log_memory(-(3 * ldt * ldt * dword + 2 * ldt * num_threads * iword), stat);

    /* Sherry added */
    SUPERLU_FREE(omp_loop_time);
    SUPERLU_FREE(full_u_cols);
    SUPERLU_FREE(blk_ldu);

    SUPERLU_FREE(lookAheadFullRow);
    SUPERLU_FREE(lookAheadStRow);
    SUPERLU_FREE(lookAhead_lptr);
    SUPERLU_FREE(lookAhead_ib);

    SUPERLU_FREE(RemainStRow);
    SUPERLU_FREE(Remain_lptr);
    SUPERLU_FREE(Remain_ib);
    SUPERLU_FREE(Remain_info);
    SUPERLU_FREE(lookAhead_L_buff);
    SUPERLU_FREE(Remain_L_buff);
    log_memory(-(3 * mrb * iword + mrb * sizeof(Remain_info_t) +
                 ldt * ldt * (num_look_aheads + 1) * dword +
                 Llu->bufmax[1] * dword),
               stat);

    SUPERLU_FREE(Ublock_info);

    /* Prepare error message - find the smallesr index i that U(i,i)==0 */
    if (*info == 0)
        *info = n + 1;
    MPI_Allreduce(info, &iinfo, 1, MPI_INT, MPI_MIN, grid->comm);
    if (iinfo == (n + 1))
        *info = 0;
    else
        *info = iinfo;

    return 0;
} /* PDGSTRF */
