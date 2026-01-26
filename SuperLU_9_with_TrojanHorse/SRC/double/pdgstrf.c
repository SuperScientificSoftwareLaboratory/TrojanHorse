/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
#include <stdio.h>
#define __USE_GNU
#include <sched.h>
#include <pthread.h>
#include <math.h>
#include "superlu_ddefs.h"
#include "gpu_api_utils.h"
#include "dscatter.c"
#include "../include/yida_thread_pool.h"
#include "cblas.h"

// #define USE_CUDA_AWARE_MPI


void trojan_horse_init(int_t nsupers);
// void trojan_horse_grouped_batched_scatter_cpu(
//     int_t nsupers,
//     int_t nrow_effective, 
//     int_t ncol_effective,
//     int_t nsupc,
//     int_t level,
//     double* bigresult_buf_d,
//     int_t* xsup_d,
//     int_t* xsup,

//     int_t bcol_nblk, // 0
//     int_t* l_nblk_bcol_prefixsum_h,
//     int_t** l_brow_idx_hh,
//     int_t** l_tile_offset_hh,
//     int_t** l_bcol_localperm_hh,
//     int_t* l_nblk_bcol_prefixsum_d,
//     int_t** l_brow_idx_dd,
//     int_t** l_tile_offset_dd,
//     int_t** l_bcol_localperm_dd,
//     double** l_bcol_val_hh,
//     int_t* metal_h,
//     int_t* l_brow_idx_h, // 1
//     int_t* l_bcol_localperm_h, // 2
//     int_t* l_tile_offset_h, // 3

//     int_t brow_nblk,
//     int_t* u_nblk_brow_prefixsum_h,
//     int_t** u_bcol_idx_hh,
//     int_t** u_tile_offset_hh,
//     int_t** u_brow_localperm_hh,
//     int_t* u_nblk_brow_prefixsum_d,
//     int_t** u_bcol_idx_dd,
//     int_t** u_tile_offset_dd,
//     int_t** u_brow_localperm_dd,
//     double** u_brow_val_hh,
//     int_t* metau_h,
//     int_t* u_bcol_idx_h,
//     int_t* u_brow_localperm_h,
//     int_t* u_tile_offset_h
// );
void trojan_horse_grouped_batched_scatter(
    int_t nsupers,
    int_t nrow_effective, 
    int_t ncol_effective,
    int_t nsupc,
    int_t level,
    double* bigresult_buf_d,
    int_t* xsup_d,
    int_t* xsup,

    int_t bcol_nblk, // 0
    int_t* l_nblk_bcol_prefixsum_h,
    int_t** l_brow_idx_hh,
    int_t** l_tile_offset_hh,
    int_t** l_bcol_localperm_hh,
    int_t* l_nblk_bcol_prefixsum_d,
    int_t** l_brow_idx_dd,
    int_t** l_tile_offset_dd,
    int_t** l_bcol_localperm_dd,
    double** l_bcol_val_dd,
    int_t* metal_h,
    int_t* l_brow_idx_d, // 1
    int_t* l_bcol_localperm_d, // 2
    int_t* l_tile_offset_d, // 3

    int_t brow_nblk,
    int_t* u_nblk_brow_prefixsum_h,
    int_t** u_bcol_idx_hh,
    int_t** u_tile_offset_hh,
    int_t** u_brow_localperm_hh,
    int_t* u_nblk_brow_prefixsum_d,
    int_t** u_bcol_idx_dd,
    int_t** u_tile_offset_dd,
    int_t** u_brow_localperm_dd,
    double** u_brow_val_dd,
    int_t* metau_h,
    int_t* u_bcol_idx_d,
    int_t* u_brow_localperm_d,
    int_t* u_tile_offset_d
);

double *stride_rs_buf = NULL;

void stride_send(void *buf, size_t row_size, size_t stride, int_t nrow, int_t start_tag, int_t tag_ub, int dest)
{
    for (int_t irow = 0; irow < nrow; irow++)
    {
        // printf("Sending %d/%d\n", irow, nrow);
        // MPI_Send(buf, row_size, MPI_CHAR, dest, tag_ub * irow + start_tag, MPI_COMM_WORLD);
        memcpy(stride_rs_buf + irow * (row_size / sizeof(double)), buf, row_size);
        buf += stride;
    }
    MPI_Send(stride_rs_buf, row_size * nrow, MPI_CHAR, dest, start_tag, MPI_COMM_WORLD);
}

void stride_recv(void *buf, size_t row_size, size_t stride, int_t nrow, int_t start_tag, int_t tag_ub, int src)
{
    MPI_Status stat;
    // for (int_t irow = 0; irow < nrow; irow++)
    // {
    //     // printf("Waiting %d/%d\n", irow, nrow);
    //     MPI_Recv(buf, row_size, MPI_CHAR, src, tag_ub * irow + start_tag, MPI_COMM_WORLD, &stat);
    //     buf += stride;
    // }
    MPI_Recv(buf, row_size * nrow, MPI_CHAR, src, start_tag, MPI_COMM_WORLD, &stat);
}

// void stride_send(void *buf, size_t row_size, size_t stride, int_t nrow, int_t start_tag, int_t tag_ub, int dest)
// {
//     for (int_t irow = 0; irow < nrow; irow++)
//     {
//         // printf("Sending %d/%d\n", irow, nrow);
//         MPI_Send(buf, row_size, MPI_CHAR, dest, tag_ub * irow + start_tag, MPI_COMM_WORLD);
//         buf += stride;
//     }
// }

// void stride_recv(void *buf, size_t row_size, size_t stride, int_t nrow, int_t start_tag, int_t tag_ub, int src)
// {
//     MPI_Status stat;
//     for (int_t irow = 0; irow < nrow; irow++)
//     {
//         // printf("Waiting %d/%d\n", irow, nrow);
//         MPI_Recv(buf, row_size, MPI_CHAR, src, tag_ub * irow + start_tag, MPI_COMM_WORLD, &stat);
//         buf += stride;
//     }
// }

void save_u_tile_to_buffer(int_t *metau_tile, double *valueu_tile, double *buffer, int_t nrow, int_t *xsup, int_t brow_of_u, int_t ncol_tile, int_t ncol_effective)
{
    memset(buffer, 0, sizeof(double) * ncol_effective * nrow);
    int_t icol = 0;
    int_t iitem = 0;
    int_t thisbrow_first_glorow = FstBlockC(brow_of_u);
    int_t nextbrow_first_glorow = FstBlockC(brow_of_u + 1);
    for (int_t colidx = 0; colidx < ncol_tile; colidx++)
    {
        int_t col_offset = metau_tile[2 + colidx] - thisbrow_first_glorow;
        if (metau_tile[2 + colidx] != nextbrow_first_glorow)
        {
            memcpy(buffer + icol * nrow + col_offset, valueu_tile + iitem, sizeof(double) * (nextbrow_first_glorow - metau_tile[2 + colidx]));
            icol++;
            iitem += (nextbrow_first_glorow - metau_tile[2 + colidx]);
        }
    }
}

void restore_buffer_to_u_tile(int_t *metau_tile, double *valueu_tile, double *buffer, int_t nrow, int_t *xsup, int_t brow_of_u, int_t ncol_tile)
{
    int_t icol = 0;
    int_t iitem = 0;
    int_t thisbrow_first_glorow = FstBlockC(brow_of_u);
    int_t nextbrow_first_glorow = FstBlockC(brow_of_u + 1);
    for (int_t colidx = 0; colidx < ncol_tile; colidx++)
    {
        int_t col_offset = metau_tile[2 + colidx] - thisbrow_first_glorow;
        if (metau_tile[2 + colidx] != nextbrow_first_glorow)
        {
            memcpy(valueu_tile + iitem, buffer + icol * nrow + col_offset, sizeof(double) * (nextbrow_first_glorow - metau_tile[2 + colidx]));
            icol++;
            iitem += (nextbrow_first_glorow - metau_tile[2 + colidx]);
        }
    }
}

int_t get_ncol_effective(int_t *metau_tile, int_t brow_of_u, int_t *xsup, int_t ncol_tile)
{
    int_t ncol_effective = 0;
    int_t nextbrow_first_glorow = FstBlockC(brow_of_u + 1);
    for (int_t colidx = 0; colidx < ncol_tile; colidx++)
    {
        if (metau_tile[2 + colidx] != nextbrow_first_glorow)
        {
            ncol_effective++;
        }
    }
    return ncol_effective;
}

#include <sys/time.h>

void timer_start(struct timeval *start)
{
    gettimeofday(start, NULL);
}

float timer_end(struct timeval *start)
{
    struct timeval end;
    gettimeofday(&end, NULL);
    return 1.0 * (end.tv_sec - start->tv_sec) + 1e-6 * (end.tv_usec - start->tv_usec);
}

void expand_double_ptr_cuda(double **ptrptr_hd, unsigned long long *len_var, unsigned long long new_len)
{
    if (new_len > *len_var)
    {
        *len_var = new_len;
        if (*ptrptr_hd)
        {
            cudaFree(*ptrptr_hd);
        }
        cudaMalloc(ptrptr_hd, sizeof(double) * new_len);
    }
}

void expand_double_ptr_pinned(double **ptrptr_hp, unsigned long long *len_var, unsigned long long new_len)
{
    if (new_len > *len_var)
    {
        *len_var = new_len;
        if (*ptrptr_hp)
        {
            cudaFreeHost(*ptrptr_hp);
        }
        cudaMallocHost(ptrptr_hp, sizeof(double) * new_len);
    }
}

double *bigcol_buf_d = NULL;
double *bigrow_buf_d = NULL;
double *bigresult_buf_d = NULL;
double *bigresult_buf_p = NULL;
unsigned long long bigcol_buf_len = 0;
unsigned long long bigrow_buf_len = 0;
unsigned long long bigresult_buf_len = 0;
unsigned long long bigresult_buf_pinned_len = 0;
cublasHandle_t handle;
cudaStream_t stream;

int_t nsupers;

int_t *recv_metau_even = NULL;
int_t recv_metau_len_even = 0;
int_t *recv_metal_even = NULL;
int_t recv_metal_len_even = 0;

int_t *recv_metau_odd = NULL;
int_t recv_metau_len_odd = 0;
int_t *recv_metal_odd = NULL;
int_t recv_metal_len_odd = 0;

#define recv_metau ((level % 2) ? recv_metau_even : recv_metau_odd)
#define recv_metau_len ((level % 2) ? recv_metau_len_even : recv_metau_len_odd)
#define recv_metal ((level % 2) ? recv_metal_even : recv_metal_odd)
#define recv_metal_len ((level % 2) ? recv_metal_len_even : recv_metal_len_odd)

int_t *xsup_d;
int_t *xsup;
dLocalLU_t *Llu;
gridinfo_t *grid_glo;
#define LYD_PGRID_P (grid_glo->nprow)
#define LYD_PGRID_Q (grid_glo->npcol)

typedef struct
{
    int_t level;
    int_t nrow_effective;
    int_t ncol_effective;
    int_t nsupc;

    int_t* l_brow_idx_d; // 1
    int_t* l_bcol_localperm_d; // 2
    int_t* l_tile_offset_d; // 3

    int_t* u_bcol_idx_d;
    int_t* u_brow_localperm_d;
    int_t* u_tile_offset_d;

    // int_t* l_brow_idx_h; // 1
    // int_t* l_bcol_localperm_h; // 2
    // int_t* l_tile_offset_h; // 3

    // int_t* u_bcol_idx_h;
    // int_t* u_brow_localperm_h;
    // int_t* u_tile_offset_h;
    
} thread_param_t;

int trojan_horse_batched_kernel(thread *_param)
{
    double one_double = 1.0;
    double zero_double = 0.0;
    int_t iam = grid_glo->iam;
    thread_param_t* param = ((thread_param_t *)(_param->param));
    int_t level = param->level;
    int_t nrow_effective = param->nrow_effective;
    int_t ncol_effective = param->ncol_effective;
    int_t nsupc = param->nsupc;

    int_t* l_brow_idx_d = param->l_brow_idx_d;
    int_t* l_bcol_localperm_d = param->l_bcol_localperm_d;
    int_t* l_tile_offset_d = param->l_tile_offset_d;

    int_t* u_bcol_idx_d = param->u_bcol_idx_d;
    int_t* u_brow_localperm_d = param->u_brow_localperm_d;
    int_t* u_tile_offset_d = param->u_tile_offset_d;

    int errid = 0;

    if ((errid = cudaDeviceSynchronize()) != cudaSuccess)
    {
        printf("cuda error 0 %d\n", errid);
        exit(1);
    }

    // timer_start(&tv_start1);
    if (nrow_effective * ncol_effective * nsupc != 0)
    {
        expand_double_ptr_cuda(&bigresult_buf_d, &bigresult_buf_len, nrow_effective * ncol_effective);
        // printf("cublasDgemm %lld %lld %lld %p %p %p\n", nrow_effective, ncol_effective, nsupc, bigcol_buf_d, bigrow_buf_d, bigresult_buf_d);
        // double bigcol_buf_h[nrow_effective*nsupc];
        // double bigrow_buf_h[nsupc*ncol_effective];
        // gpuMemcpy(bigcol_buf_h, bigcol_buf_d, sizeof(double)*nrow_effective*nsupc, cudaMemcpyDeviceToHost);
        // gpuMemcpy(bigrow_buf_h, bigrow_buf_d, sizeof(double)*nsupc*ncol_effective, cudaMemcpyDeviceToHost);

        // printf("bigcol %d %dx%d\n", level, nrow_effective, nsupc);
        // for(int i=0;i<nrow_effective;i++){
        //     for(int j=0;j<nsupc;j++){
        //         printf("%.2lf ", bigcol_buf_h[j*nrow_effective+i]);
        //     }
        //     printf("\n");
        // }

        // printf("bigrow %d %dx%d\n", level, nsupc, ncol_effective);
        // for(int i=0;i<nsupc;i++){
        //     for(int j=0;j<ncol_effective;j++){
        //         printf("%.2lf ", bigrow_buf_h[j*nsupc+i]);
        //     }
        //     printf("\n");
        // }

        cublasDgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            nrow_effective, ncol_effective, nsupc,
            &one_double,
            bigcol_buf_d, nrow_effective,
            bigrow_buf_d, nsupc,
            &zero_double,
            bigresult_buf_d, nrow_effective);
        // expand_double_ptr_pinned(&bigresult_buf_p, &bigresult_buf_pinned_len, nrow_effective * ncol_effective);
        // cudaMemcpyAsync(bigresult_buf_p, bigresult_buf_d, sizeof(double) * nrow_effective * ncol_effective, cudaMemcpyDeviceToHost, stream);
    }
    // time_tmp3 += timer_end(&tv_start1);

    if ((errid = cudaDeviceSynchronize()) != cudaSuccess)
    {
        printf("cuda error 1 %d\n", errid);
        exit(1);
    }

    cudaDeviceSynchronize();

    // SSSSM, update
    // timer_start(&tv_start1);
    trojan_horse_grouped_batched_scatter(
        nsupers,
        nrow_effective,
        ncol_effective,
        nsupc,
        level,
        bigresult_buf_d,
        xsup_d,
        xsup,

        recv_metal[0],
        Llu->l_nblk_bcol_prefixsum_h,
        Llu->l_brow_idx_hh,
        Llu->l_tile_offset_hh,
        Llu->l_bcol_localperm_hh,
        Llu->l_nblk_bcol_prefixsum_d,
        Llu->l_brow_idx_dd,
        Llu->l_tile_offset_dd,
        Llu->l_bcol_localperm_dd,
        Llu->l_bcol_val_dd,
        recv_metal,
        l_brow_idx_d,
        l_bcol_localperm_d,
        l_tile_offset_d,
        
        recv_metau[0],
        Llu->u_nblk_brow_prefixsum_h,
        Llu->u_bcol_idx_hh,
        Llu->u_tile_offset_hh,
        Llu->u_brow_localperm_hh,
        Llu->u_nblk_brow_prefixsum_d,
        Llu->u_bcol_idx_dd,
        Llu->u_tile_offset_dd,
        Llu->u_brow_localperm_dd,
        Llu->u_brow_val_dd,
        recv_metau,
        u_bcol_idx_d,
        u_brow_localperm_d,
        u_tile_offset_d
    );

    if ((errid = cudaDeviceSynchronize()) != cudaSuccess)
    {
        printf("cuda error 2 %d\n", errid);
        exit(1);
    }

    cudaDeviceSynchronize();

    if (((level + 2) < nsupers) && (((level + 2) % LYD_PGRID_Q) == (iam % LYD_PGRID_Q)) && Llu->Lrowind_bc_ptr[LBj((level + 2), grid_glo)])
    // if (((level + 2) < nsupers) && Llu->Lrowind_bc_ptr[LBj((level + 2), grid_glo)])
    {
        gpuMemcpyAsync(
            Llu->Lnzval_bc_ptr[LBj((level + 2), grid_glo)],
            Llu->l_bcol_val_hd[level + 2],
            sizeof(double) * Llu->Lrowind_bc_ptr[LBj((level + 2), grid_glo)][1] * SuperSize(level + 2),
            gpuMemcpyDeviceToHost, stream);

        // printf("#%d Download %d*double level=%d\n", iam, Llu->Lrowind_bc_ptr[LBj((level + 2), grid_glo)][1] * SuperSize(level + 2), level+2);
        // int nrow_effective = Llu->Lrowind_bc_ptr[LBj((level + 2), grid_glo)][1];
        // int nsupc = SuperSize(level + 2);
        // printf("bigcol %d %dx%d\n", level+2, nrow_effective, nsupc);
        // for(int i=0;i<nrow_effective;i++){
        //     for(int j=0;j<nsupc;j++){
        //         printf("%.2lf ", Llu->Lnzval_bc_ptr[LBj((level + 2), grid_glo)][j*nrow_effective+i]);
        //     }
        //     printf("\n");
        // }
    }

    if (((level + 2) < nsupers) && (((level + 2) % LYD_PGRID_P) == (iam / LYD_PGRID_Q)) && Llu->Ufstnz_br_ptr[LBi((level + 2), grid_glo)])
    // if (((level + 2) < nsupers) && Llu->Ufstnz_br_ptr[LBi((level + 2), grid_glo)])
    {
        gpuMemcpyAsync(
            Llu->Unzval_br_ptr[LBi((level + 2), grid_glo)],
            Llu->u_brow_val_hd[level + 2],
            sizeof(double) * Llu->Ufstnz_br_ptr[LBi((level + 2), grid_glo)][1],
            gpuMemcpyDeviceToHost, stream);
    }

    if ((errid = cudaDeviceSynchronize()) != cudaSuccess)
    {
        printf("cuda error 3 %d\n", errid);
        exit(1);
    }
}

cublasStatus_t cublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double *alpha,
                           const double *A, int lda,
                           const double *B, int ldb,
                           const double *beta,
                           double *C, int ldc);

/// @brief
/// @param options
/// @param m
/// @param n
/// @param anorm
/// @param LUstruct
/// @param grid
/// @param stat
/// @param info
/// @return
int_t pdgstrf(
    superlu_dist_options_t *options, int m, int n, double anorm,
    dLUstruct_t *LUstruct, gridinfo_t *grid, SuperLUStat_t *stat, int *info)
{
    grid_glo = grid;

    // #define LYD_PGRID_PROW(brow) ((brow) / LYD_PGRID_Q)
    // #define LYD_PGRID_PCOL(bcol) ((bcol) % LYD_PGRID_Q)
    // #define LYD_PGRID_RANK(brow, bcol) (brow * LYD_PGRID_Q + bcol)

    float time_init = 0;

    float time_getrf_all = 0;
    float time_getrf_kernel = 0;
    float time_getrf_update = 0;
    float time_getrf_update_kernel = 0;
    float time_getrf_send = 0;
    float time_getrf_recv = 0;

    float time_tstrf_all = 0;
    float time_tstrf_kernel = 0;
    float time_tstrf_update = 0;
    float time_tstrf_update_kernel = 0;
    float time_tstrf_send = 0;
    float time_tstrf_recv = 0;

    float time_gessm_all = 0;
    float time_gessm_kernel = 0;
    float time_gessm_update = 0;
    float time_gessm_update_kernel = 0;
    float time_gessm_send = 0;
    float time_gessm_recv = 0;

    float time_divide_u = 0;
    float time_divide_l = 0;

    float time_tmp1 = 0;
    float time_tmp2 = 0;
    float time_tmp3 = 0;
    float time_tmp4 = 0;
    float time_tmp5 = 0;
    float time_tmp6 = 0;
    float time_tmp7 = 0;

    struct timeval tv_start1, tv_start2, tv_start3;

    int flying_virtual_tid = 0;

    int_t iam = grid->iam;
    int_t nproc = LYD_PGRID_P * LYD_PGRID_Q;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    Llu = LUstruct->Llu;

    cuda_device = iam;
    int cuda_dev_cnt;
    cudaGetDeviceCount(&cuda_dev_cnt);
    cudaSetDevice(cuda_device % cuda_dev_cnt);

    xsup = Glu_persist->xsup;
    nsupers = Glu_persist->supno[n - 1] + 1;
    int_t superlu_relax = sp_ienv_dist(2, options);
    int_t superlu_maxsup = sp_ienv_dist(3, options);
    int_t **Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    double **Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    int_t **Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    double **Unzval_br_ptr = Llu->Unzval_br_ptr;
    int *blas_ipiv_1base = SUPERLU_MALLOC(superlu_maxsup * sizeof(int));
    for (int i = 0; i < superlu_maxsup; i++)
    {
        blas_ipiv_1base[i] = 0;
    }

    cudaMalloc(&xsup_d, sizeof(int_t) * (nsupers + 1));
    cudaMemcpy(xsup_d, xsup, sizeof(int_t) * (nsupers + 1), cudaMemcpyHostToDevice);
    char *send_flag = SUPERLU_MALLOC(sizeof(char) * nproc);
    double *pivot_ptr = NULL;
    int_t pivot_lda = 0;
    double *pivot_buf = SUPERLU_MALLOC(sizeof(double) * superlu_maxsup * superlu_maxsup);

    double *recv_tempu = NULL;
    int_t recv_tempu_len = 0;
    double *recv_templ = NULL;
    int_t recv_templ_len = 0;
    int_t ssssm_result_cap = superlu_maxsup * superlu_maxsup;
    double *ssssm_result = SUPERLU_MALLOC(sizeof(double) * ssssm_result_cap);
    double *u_tile_buffer = SUPERLU_MALLOC(sizeof(double) * superlu_maxsup * superlu_maxsup);
    double *a_tile_buffer = SUPERLU_MALLOC(sizeof(double) * superlu_maxsup * superlu_maxsup);
    int_t *indirect_l_1 = SUPERLU_MALLOC(sizeof(int_t) * superlu_maxsup);
    int_t *indirect_l_2 = SUPERLU_MALLOC(sizeof(int_t) * superlu_maxsup);
    int_t *indirect_u_1 = SUPERLU_MALLOC(sizeof(int_t) * superlu_maxsup);
    int_t *indirect_u_2 = SUPERLU_MALLOC(sizeof(int_t) * superlu_maxsup);
    double thresh = smach_dist("Epsilon") * anorm;
    stride_rs_buf = SUPERLU_MALLOC(sizeof(double) * superlu_maxsup * superlu_maxsup);

    double *l_fullrow_buf = SUPERLU_MALLOC(sizeof(double) * superlu_maxsup * superlu_maxsup);

    int_t* l_brow_idx_even_h; // 1 <nsupers
    int_t* l_bcol_localperm_even_h; // 2 <n
    int_t* l_tile_offset_even_h; // 3 <(nsupers+1)
    int_t* u_bcol_idx_even_h;
    int_t* u_brow_localperm_even_h;
    int_t* u_tile_offset_even_h;
    l_brow_idx_even_h = malloc(sizeof(int_t)*((nsupers) + (n) + (nsupers+1))*2);
    l_bcol_localperm_even_h = l_brow_idx_even_h + nsupers;
    l_tile_offset_even_h = l_bcol_localperm_even_h + n;
    u_bcol_idx_even_h = l_tile_offset_even_h + (nsupers+1);
    u_brow_localperm_even_h = u_bcol_idx_even_h + nsupers;
    u_tile_offset_even_h = u_brow_localperm_even_h + n;
    
    int_t* l_brow_idx_even_d; // 1 <nsupers
    int_t* l_bcol_localperm_even_d; // 2 <n
    int_t* l_tile_offset_even_d; // 3 <(nsupers+1)
    int_t* u_bcol_idx_even_d;
    int_t* u_brow_localperm_even_d;
    int_t* u_tile_offset_even_d;
    cudaMalloc(&l_brow_idx_even_d, sizeof(int_t)*((nsupers) + (n) + (nsupers+1))*2);
    l_bcol_localperm_even_d = l_brow_idx_even_d + nsupers;
    l_tile_offset_even_d = l_bcol_localperm_even_d + n;
    u_bcol_idx_even_d = l_tile_offset_even_d + (nsupers+1);
    u_brow_localperm_even_d = u_bcol_idx_even_d + nsupers;
    u_tile_offset_even_d = u_brow_localperm_even_d + n;

    int_t* l_brow_idx_odd_h; // 1 <nsupers
    int_t* l_bcol_localperm_odd_h; // 2 <n
    int_t* l_tile_offset_odd_h; // 3 <(nsupers+1)
    int_t* u_bcol_idx_odd_h;
    int_t* u_brow_localperm_odd_h;
    int_t* u_tile_offset_odd_h;
    l_brow_idx_odd_h = malloc(sizeof(int_t)*((nsupers) + (n) + (nsupers+1))*2);
    l_bcol_localperm_odd_h = l_brow_idx_odd_h + nsupers;
    l_tile_offset_odd_h = l_bcol_localperm_odd_h + n;
    u_bcol_idx_odd_h = l_tile_offset_odd_h + (nsupers+1);
    u_brow_localperm_odd_h = u_bcol_idx_odd_h + nsupers;
    u_tile_offset_odd_h = u_brow_localperm_odd_h + n;

    int_t* l_brow_idx_odd_d; // 1 <nsupers
    int_t* l_bcol_localperm_odd_d; // 2 <n
    int_t* l_tile_offset_odd_d; // 3 <(nsupers+1)
    int_t* u_bcol_idx_odd_d;
    int_t* u_brow_localperm_odd_d;
    int_t* u_tile_offset_odd_d;
    cudaMalloc(&l_brow_idx_odd_d, sizeof(int_t)*((nsupers) + (n) + (nsupers+1))*2);
    l_bcol_localperm_odd_d = l_brow_idx_odd_d + nsupers;
    l_tile_offset_odd_d = l_bcol_localperm_odd_d + n;
    u_bcol_idx_odd_d = l_tile_offset_odd_d + (nsupers+1);
    u_brow_localperm_odd_d = u_bcol_idx_odd_d + nsupers;
    u_tile_offset_odd_d = u_brow_localperm_odd_d + n;

    cublasCreate(&handle);
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    trojan_horse_init(nsupers);
    bind_to_core(2 * iam);

    Thread_Pool *trojan_horse_thread_pool = create_pool(1, 2 * iam + 1);
    thread_param_t *thread_param = SUPERLU_MALLOC(sizeof(thread_param_t));
    thread_param->level = 0;

    if (!iam)
    {
        printf("nsupers=%d\n", nsupers);
    }

    // Main loop
    for (int_t level = 0; level < nsupers; level++)
    {

        // printf("level=%lld\n", level);
        // printf("%d 1\n", level);

        int_t nsupc = SuperSize(level);
        int_t level_local_col = LBj(level, grid);
        int_t level_local_row = LBi(level, grid);

        timer_start(&tv_start1);
        // GETRF
        if (iam == (level % LYD_PGRID_P) * LYD_PGRID_Q + (level % LYD_PGRID_Q))
        {
            double *lusup = Llu->Lnzval_bc_ptr[level_local_col];
            int_t nblk = Llu->Lrowind_bc_ptr[level_local_col][0];
            int_t lda = Llu->Lrowind_bc_ptr[level_local_col][1];
            double *getrf_block = &lusup[0];
            int kernel_info = 0;

            timer_start(&tv_start2);
            int cols_left = nsupc;
            int_t luptr = 0;
            int_t u_diag_cnt = 0;
            int_t ld_ujrow = lda;
            double *ublk_ptr = getrf_block;
            double *ujrow = getrf_block;
            double alpha = -1;
            for (int_t j = 0; j < nsupc; ++j) /* for each column in panel */
            {
                /* Diagonal pivot */
                int_t i = luptr;

                for (int_t l = 0; l < cols_left; ++l, i += ld_ujrow, ++u_diag_cnt)
                {
                    int_t st = j * ld_ujrow + j;
                    ublk_ptr[st + l * ld_ujrow] = lusup[i]; /* copy one row of U */
                }

                double temp;
                temp = 1.0 / ujrow[0];
                for (int_t i = luptr + 1; i < luptr - j + nsupc; ++i)
                    lusup[i] *= temp;

                /* Rank-1 update of the trailing submatrix. */
                if (--cols_left)
                {
                    /*following must be int*/
                    int l = nsupc - j - 1;
                    /* Rank-1 update */
                    superlu_dger(
                        l, cols_left, alpha,
                        &lusup[luptr + 1], 1,
                        &ujrow[ld_ujrow], ld_ujrow,
                        &lusup[luptr + ld_ujrow + 1], ld_ujrow);
                }

                ujrow = ujrow + ld_ujrow + 1; /* move to next row of U */
                luptr += ld_ujrow + 1;        /* move to next column */

            } /* for column j ...  first loop */
            time_getrf_kernel += timer_end(&tv_start2);
            time_getrf_all += timer_end(&tv_start2);

            // printf("#%d GETRF (%d, %d) n=%d stride=%d\n", iam, level, level, nsupc, lda);

            memset(send_flag, 0, sizeof(char) * nproc);
            for (int remote_rank = 0; remote_rank < nproc; remote_rank++)
            {
                if (remote_rank / LYD_PGRID_Q == iam / LYD_PGRID_Q)
                {
                    send_flag[remote_rank] = 1;
                }
                if (remote_rank % LYD_PGRID_Q == iam % LYD_PGRID_Q)
                {
                    send_flag[remote_rank] = 1;
                }
                if (remote_rank == iam)
                {
                    send_flag[remote_rank] = 0;
                }
            }
            for (int_t irow = 0; irow < nsupc; irow++)
            {
                memcpy(stride_rs_buf + irow * nsupc, getrf_block + irow * lda, sizeof(double) * nsupc);
            }
            for (int remote_rank = 0; remote_rank < nproc; remote_rank++)
            {
                if (send_flag[remote_rank])
                {
                    // printf("#%d stride_send(row_size=%d, stride=%d, nrow=%d, %d->%d)\n", iam, nsupc, lda, nsupc, iam, remote_rank);
                    // stride_send(getrf_block, sizeof(double) * nsupc, sizeof(double) * lda, nsupc, 0, 10, remote_rank);
                    MPI_Send(stride_rs_buf, sizeof(double) * nsupc * nsupc, MPI_CHAR, remote_rank, 0, MPI_COMM_WORLD);
                }
            }
            pivot_ptr = getrf_block;
            pivot_lda = lda;
        }
        else
        {
            int pivot_rank = (level % LYD_PGRID_P) * LYD_PGRID_Q + (level % LYD_PGRID_Q);
            if ((pivot_rank / LYD_PGRID_Q == iam / LYD_PGRID_Q) || (pivot_rank % LYD_PGRID_Q == iam % LYD_PGRID_Q))
            {
                // stride_recv(pivot_buf, sizeof(double) * nsupc, sizeof(double) * nsupc, nsupc, 0, 10, pivot_rank);
                MPI_Status stat;
                MPI_Recv(pivot_buf, sizeof(double) * nsupc * nsupc, MPI_CHAR, pivot_rank, 0, MPI_COMM_WORLD, &stat);
                // printf("#%d stride_recv(row_size=%d, stride=%d, nrow=%d, %d->%d)\n", iam, nsupc, nsupc, nsupc, pivot_rank, iam);
            }
            pivot_ptr = pivot_buf;
            pivot_lda = nsupc;
        }
        time_tmp7 += timer_end(&tv_start1);

        int_t lda = -1;
        int_t metal_len = -1;
        int_t meta_idx = -1;

        // TSTRF
        timer_start(&tv_start1);
        int_t nrow_effective = 0;
        int_t *metal = Lrowind_bc_ptr[level_local_col];
        double *valuel = Lnzval_bc_ptr[level_local_col];
        if (iam % LYD_PGRID_Q != level % LYD_PGRID_Q)
        {
            metal = NULL;
            valuel = NULL;
        }
        if (metal)
        {
            int_t nblk_col = metal[0];
            nrow_effective = metal[1];
            metal_len = 2 + 2 * nblk_col + nrow_effective;
            meta_idx = 2;
            lda = nrow_effective;


            timer_start(&tv_start2);
            if (metal[meta_idx] == level)
            {
                // printf("bigcol %d %dx%d *\n", level, nrow_effective - metal[meta_idx + 1], nsupc);
                // for(int i=0;i<nrow_effective - metal[meta_idx + 1];i++){
                //     for(int j=0;j<nsupc;j++){
                //         printf("%.2lf ", valuel[metal[meta_idx + 1] + j*nrow_effective+i]);
                //     }
                //     printf("\n");
                // }
                cblas_dtrsm(
                    CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                    nsupc, nrow_effective - metal[meta_idx + 1], 1.0, pivot_ptr, pivot_lda, valuel + metal[meta_idx + 1], nrow_effective);
            }
            else
            {
                // printf("bigcol %d %dx%d\n", level, nrow_effective, nsupc);
                // for(int i=0;i<nrow_effective;i++){
                //     for(int j=0;j<nsupc;j++){
                //         printf("%.2lf ", valuel[j*nrow_effective+i]);
                //     }
                //     printf("\n");
                // }
                cblas_dtrsm(
                    CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                    nsupc, nrow_effective, 1.0, pivot_ptr, pivot_lda, valuel, nrow_effective);
            }
            time_tstrf_kernel += timer_end(&tv_start2);
        }
        time_tstrf_all += timer_end(&tv_start1);

        // GESSM
        timer_start(&tv_start1);
        int_t *metau = Ufstnz_br_ptr[level_local_row];
        double *valueu = Unzval_br_ptr[level_local_row];
        int_t ncol_effective = 0;
        if (iam / LYD_PGRID_Q != level % LYD_PGRID_P)
        {
            metau = NULL;
            valueu = NULL;
        }
        if (metau)
        {
            int_t nzblk_row = metau[0];
            int_t nnz_row = metau[1];

            ncol_effective = nnz_row / nsupc;
            timer_start(&tv_start2);
            cblas_dtrsm(
                CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                nsupc, ncol_effective, 1.0, pivot_ptr, pivot_lda, valueu, nsupc);
            time_gessm_kernel += timer_end(&tv_start2);
        }
        time_gessm_all += timer_end(&tv_start1);

        timer_start(&tv_start1);
        // Send U
        int_t pcol = iam % LYD_PGRID_Q;
        int_t prow = iam / LYD_PGRID_Q;
        if (metau)
        {
            int_t nzblk_row = metau[0];
            int_t nnz_row = metau[1];
            int_t metau_len = metau[2];
            // printf("%p\n", metau + 2);
            if (prow == level % LYD_PGRID_P)
            {
                // printf("#%d own panel level=%d and have metau\n", iam, level);
                for (int_t target_prow = 0; target_prow < LYD_PGRID_P; target_prow++)
                {
                    if (target_prow != prow)
                    {
                        MPI_Request req;
                        int target_rank = target_prow * LYD_PGRID_Q + pcol;
                        MPI_Send(metau, metau_len, mpi_int_t, target_rank, 0, MPI_COMM_WORLD);
                        MPI_Send(valueu, nnz_row, MPI_DOUBLE, target_rank, 1, MPI_COMM_WORLD);

                        #ifdef USE_CUDA_AWARE_MPI
                        MPI_Send(Llu->u_tile_offset_hd[level], (metau[0]+1), mpi_int_t, target_rank, 201, MPI_COMM_WORLD);
                        // cudaError_t error = cudaGetLastError();
                        // printf("CUDA error @201: %s\n", cudaGetErrorString(error));

                        MPI_Send(Llu->u_brow_localperm_hd[level], metau[1]/nsupc, mpi_int_t, target_rank, 202, MPI_COMM_WORLD);
                        // error = cudaGetLastError();
                        // printf("CUDA error @202: %s\n", cudaGetErrorString(error));

                        MPI_Send(Llu->u_bcol_idx_hd[level], metau[0], mpi_int_t, target_rank, 203, MPI_COMM_WORLD);
                        // error = cudaGetLastError();
                        // printf("CUDA error @203: %s\n", cudaGetErrorString(error));

                        #else

                        cudaMemcpy(Llu->u_tile_offset_hh[level], Llu->u_tile_offset_hd[level], sizeof(mpi_int_t) * (metau[0]+1), cudaMemcpyDeviceToHost);
                        MPI_Send(Llu->u_tile_offset_hh[level], (metau[0]+1), mpi_int_t, target_rank, 201, MPI_COMM_WORLD);

                        cudaMemcpy(Llu->u_brow_localperm_hh[level], Llu->u_brow_localperm_hd[level], sizeof(mpi_int_t) * (metau[1]/nsupc), cudaMemcpyDeviceToHost);
                        MPI_Send(Llu->u_brow_localperm_hh[level], metau[1]/nsupc, mpi_int_t, target_rank, 202, MPI_COMM_WORLD);
                        
                        cudaMemcpy(Llu->u_bcol_idx_hh[level], Llu->u_bcol_idx_hd[level], sizeof(mpi_int_t) * metau[0], cudaMemcpyDeviceToHost);
                        MPI_Send(Llu->u_bcol_idx_hh[level], metau[0], mpi_int_t, target_rank, 203, MPI_COMM_WORLD);

                        #endif
                        
                        // printf("metau %d %d*%d\n", level, nsupc, nnz_row/nsupc);
                        // for(int i=0;i<nsupc;i++){
                        //     for(int j=0;j<nnz_row/nsupc;j++){
                        //         printf("%.2lf ", valueu[j*nsupc+i]);
                        //     }
                        //     printf("\n");
                        // }
                    }
                    else
                    {
                        if (metau_len > recv_metau_len)
                        {
                            if (recv_metau)
                            {
                                SUPERLU_FREE(recv_metau);
                            }
                            if (level % 2)
                            {
                                recv_metau_even = SUPERLU_MALLOC(sizeof(int_t) * metau_len);
                                recv_metau_len_even = metau_len;
                            }
                            else
                            {
                                recv_metau_odd = SUPERLU_MALLOC(sizeof(int_t) * metau_len);
                                recv_metau_len_odd = metau_len;
                            }
                        }
                        memcpy(recv_metau, metau, sizeof(int_t) * metau_len);
                        if (nnz_row > recv_tempu_len)
                        {
                            if (recv_tempu)
                            {
                                SUPERLU_FREE(recv_tempu);
                            }
                            recv_tempu = SUPERLU_MALLOC(sizeof(double) * nnz_row);
                            recv_tempu_len = nnz_row;
                        }
                        memcpy(recv_tempu, valueu, sizeof(double) * nnz_row);
                    }
                }
            }
        }
        else
        {
            if (prow == level % LYD_PGRID_P)
            {
                // printf("#%d own panel level=%d but do not have metau\n", iam, level);
                for (int_t target_prow = 0; target_prow < LYD_PGRID_P; target_prow++)
                {
                    if (target_prow != iam / LYD_PGRID_Q)
                    {
                        MPI_Request req;
                        MPI_Send(metau, 0, mpi_int_t, target_prow * LYD_PGRID_Q + pcol, 0, MPI_COMM_WORLD);
                        // MPI_Send(valueu, 0, MPI_DOUBLE, target_prow * LYD_PGRID_Q + pcol, 1, MPI_COMM_WORLD);
                    }
                    else
                    {
                        if (3 > recv_metau_len)
                        {
                            if (recv_metau)
                            {
                                SUPERLU_FREE(recv_metau);
                            }
                            if (level % 2)
                            {
                                recv_metau_even = SUPERLU_MALLOC(sizeof(int_t) * 3);
                                recv_metau_len_even = 3;
                            }
                            else
                            {
                                recv_metau_odd = SUPERLU_MALLOC(sizeof(int_t) * 3);
                                recv_metau_len_odd = 3;
                            }
                        }
                        recv_metau[0] = 0;
                        recv_metau[1] = 0;
                        recv_metau[2] = 3;
                    }
                }
            }
        }
        if (prow != level % LYD_PGRID_P)
        {
            // printf("#%d do not own panel level=%d\n", iam, level);
            MPI_Status mpi_stat;
            int count = 0;

            timer_start(&tv_start2);
            MPI_Probe((level % LYD_PGRID_P) * LYD_PGRID_Q + pcol, 0, MPI_COMM_WORLD, &mpi_stat);
            time_tmp5 += timer_end(&tv_start2);
            MPI_Get_count(&mpi_stat, mpi_int_t, &count);
            if (count != 0)
            {
                if (count > recv_metau_len)
                {
                    if (recv_metau)
                    {
                        SUPERLU_FREE(recv_metau);
                    }
                    if (level % 2)
                    {
                        recv_metau_even = SUPERLU_MALLOC(sizeof(int_t) * count);
                        recv_metau_len_even = count;
                    }
                    else
                    {
                        recv_metau_odd = SUPERLU_MALLOC(sizeof(int_t) * count);
                        recv_metau_len_odd = count;
                    }
                    // recv_metau = SUPERLU_MALLOC(sizeof(int_t) * count);
                    // recv_metau_len = count;
                }
                // timer_start(&tv_start2);
                MPI_Recv(recv_metau, count, mpi_int_t, (level % LYD_PGRID_P) * LYD_PGRID_Q + pcol, 0, MPI_COMM_WORLD, &mpi_stat);
                ncol_effective = recv_metau[1]/nsupc;
                // time_tmp5 += timer_end(&tv_start2);

                timer_start(&tv_start2);
                MPI_Probe(((level % LYD_PGRID_P) % LYD_PGRID_P) * LYD_PGRID_Q + pcol, 1, MPI_COMM_WORLD, &mpi_stat);
                time_tmp5 += timer_end(&tv_start2);
                MPI_Get_count(&mpi_stat, MPI_DOUBLE, &count);
                if (count > recv_tempu_len)
                {
                    if (recv_tempu)
                    {
                        SUPERLU_FREE(recv_tempu);
                    }
                    recv_tempu = SUPERLU_MALLOC(sizeof(double) * count);
                    recv_tempu_len = count;
                }
                int fetch_rank = (level % LYD_PGRID_P) * LYD_PGRID_Q + pcol;
                MPI_Recv(recv_tempu, count, MPI_DOUBLE, fetch_rank, 1, MPI_COMM_WORLD, &mpi_stat);
                #ifdef USE_CUDA_AWARE_MPI
                if(level%2 == 1){ // odd
                    MPI_Recv(u_tile_offset_odd_d, (recv_metau[0]+1), mpi_int_t, fetch_rank, 201, MPI_COMM_WORLD, &mpi_stat);
                    MPI_Recv(u_brow_localperm_odd_d, recv_metau[1]/nsupc, mpi_int_t, fetch_rank, 202, MPI_COMM_WORLD, &mpi_stat);
                    MPI_Recv(u_bcol_idx_odd_d, recv_metau[0], mpi_int_t, fetch_rank, 203, MPI_COMM_WORLD, &mpi_stat);
                }else{ // even
                    MPI_Recv(u_tile_offset_even_d, (recv_metau[0]+1), mpi_int_t, fetch_rank, 201, MPI_COMM_WORLD, &mpi_stat);
                    MPI_Recv(u_brow_localperm_even_d, recv_metau[1]/nsupc, mpi_int_t, fetch_rank, 202, MPI_COMM_WORLD, &mpi_stat);
                    MPI_Recv(u_bcol_idx_even_d, recv_metau[0], mpi_int_t, fetch_rank, 203, MPI_COMM_WORLD, &mpi_stat);
                }
                #else
                if(level%2 == 1){ // odd
                    MPI_Recv(u_tile_offset_odd_h, (recv_metau[0]+1), mpi_int_t, fetch_rank, 201, MPI_COMM_WORLD, &mpi_stat);
                    cudaMemcpy(u_tile_offset_odd_d, u_tile_offset_odd_h, sizeof(mpi_int_t) * (recv_metau[0]+1), cudaMemcpyHostToDevice);
                    MPI_Recv(u_brow_localperm_odd_h, recv_metau[1]/nsupc, mpi_int_t, fetch_rank, 202, MPI_COMM_WORLD, &mpi_stat);
                    cudaMemcpy(u_brow_localperm_odd_d, u_brow_localperm_odd_h, sizeof(mpi_int_t) * (recv_metau[1]/nsupc), cudaMemcpyHostToDevice);
                    MPI_Recv(u_bcol_idx_odd_h, recv_metau[0], mpi_int_t, fetch_rank, 203, MPI_COMM_WORLD, &mpi_stat);
                    cudaMemcpy(u_bcol_idx_odd_d, u_bcol_idx_odd_h, sizeof(mpi_int_t) * recv_metau[0], cudaMemcpyHostToDevice);
                }else{ // even
                    MPI_Recv(u_tile_offset_even_h, (recv_metau[0]+1), mpi_int_t, fetch_rank, 201, MPI_COMM_WORLD, &mpi_stat);
                    cudaMemcpy(u_tile_offset_even_d, u_tile_offset_even_h, sizeof(mpi_int_t) * (recv_metau[0]+1), cudaMemcpyHostToDevice);
                    MPI_Recv(u_brow_localperm_even_h, recv_metau[1]/nsupc, mpi_int_t, fetch_rank, 202, MPI_COMM_WORLD, &mpi_stat);
                    cudaMemcpy(u_brow_localperm_even_d, u_brow_localperm_even_h, sizeof(mpi_int_t) * (recv_metau[1]/nsupc), cudaMemcpyHostToDevice);
                    MPI_Recv(u_bcol_idx_even_h, recv_metau[0], mpi_int_t, fetch_rank, 203, MPI_COMM_WORLD, &mpi_stat);
                    cudaMemcpy(u_bcol_idx_even_d, u_bcol_idx_even_h, sizeof(mpi_int_t) * recv_metau[0], cudaMemcpyHostToDevice);
                }
                #endif
            }
            else
            {
                if (3 > recv_metau_len)
                {
                    if (recv_metau)
                    {
                        SUPERLU_FREE(recv_metau);
                    }
                    // recv_metau = SUPERLU_MALLOC(sizeof(int_t) * 3);
                    // recv_metau_len = 3;
                    if (level % 2)
                    {
                        recv_metau_even = SUPERLU_MALLOC(sizeof(int_t) * 3);
                        recv_metau_len_even = 3;
                    }
                    else
                    {
                        recv_metau_odd = SUPERLU_MALLOC(sizeof(int_t) * 3);
                        recv_metau_len_odd = 3;
                    }
                }
                // timer_start(&tv_start2);
                MPI_Recv(recv_metau, count, mpi_int_t, (level % LYD_PGRID_P) * LYD_PGRID_Q + pcol, 0, MPI_COMM_WORLD, &mpi_stat);
                // time_tmp5 += timer_end(&tv_start2);
                recv_metau[0] = 0;
                recv_metau[1] = 0;
                recv_metau[2] = 3;
            }
        }

        // Send L
        if (metal)
        {
            if (pcol == level % LYD_PGRID_Q)
            {
                for (int_t target_pcol = 0; target_pcol < LYD_PGRID_Q; target_pcol++)
                {
                    if (target_pcol != iam % LYD_PGRID_Q)
                    {
                        MPI_Request req;
                        int target_rank = prow * LYD_PGRID_Q + target_pcol;
                        MPI_Send(metal, metal_len, mpi_int_t, target_rank, 2, MPI_COMM_WORLD);
                        MPI_Send(valuel, lda * nsupc, MPI_DOUBLE, target_rank, 3, MPI_COMM_WORLD);

                        #ifdef USE_CUDA_AWARE_MPI
                        MPI_Send(Llu->l_tile_offset_hd[level], (metal[0]+1), mpi_int_t, target_rank, 101, MPI_COMM_WORLD);
                        // cudaError_t error = cudaGetLastError();
                        // if(error)
                        // printf("CUDA error @101: %s\n", cudaGetErrorString(error));

                        MPI_Send(Llu->l_bcol_localperm_hd[level], metal[1], mpi_int_t, target_rank, 102, MPI_COMM_WORLD);
                        // error = cudaGetLastError();
                        // if(error)
                        // printf("CUDA error @102: %s\n", cudaGetErrorString(error));

                        MPI_Send(Llu->l_brow_idx_hd[level], metal[0], mpi_int_t, target_rank, 103, MPI_COMM_WORLD);
                        // error = cudaGetLastError();
                        // if(error)
                        // printf("CUDA error @103: %s\n", cudaGetErrorString(error));
                        #else

                        cudaMemcpy(Llu->l_tile_offset_hh[level], Llu->l_tile_offset_hd[level], sizeof(mpi_int_t) * (metal[0]+1), cudaMemcpyDeviceToHost);
                        MPI_Send(Llu->l_tile_offset_hh[level], (metal[0]+1), mpi_int_t, target_rank, 101, MPI_COMM_WORLD);

                        cudaMemcpy(Llu->l_bcol_localperm_hh[level], Llu->l_bcol_localperm_hd[level], sizeof(mpi_int_t) * metal[1], cudaMemcpyDeviceToHost);
                        MPI_Send(Llu->l_bcol_localperm_hh[level], metal[1], mpi_int_t, target_rank, 102, MPI_COMM_WORLD);

                        cudaMemcpy(Llu->l_brow_idx_hh[level], Llu->l_brow_idx_hd[level], sizeof(mpi_int_t) * metal[0], cudaMemcpyDeviceToHost);
                        MPI_Send(Llu->l_brow_idx_hh[level], metal[0], mpi_int_t, target_rank, 103, MPI_COMM_WORLD);

                        #endif
                    }
                    else
                    {
                        if (metal_len > recv_metal_len)
                        {
                            if (recv_metal)
                            {
                                SUPERLU_FREE(recv_metal);
                            }
                            // recv_metal = SUPERLU_MALLOC(sizeof(int_t) * metal_len);
                            // recv_metal_len = metal_len;
                            if (level % 2)
                            {
                                recv_metal_even = SUPERLU_MALLOC(sizeof(int_t) * metal_len);
                                recv_metal_len_even = metal_len;
                            }
                            else
                            {
                                recv_metal_odd = SUPERLU_MALLOC(sizeof(int_t) * metal_len);
                                recv_metal_len_odd = metal_len;
                            }
                        }
                        memcpy(recv_metal, metal, sizeof(int_t) * metal_len);
                        if (lda * nsupc > recv_templ_len)
                        {
                            if (recv_templ)
                            {
                                SUPERLU_FREE(recv_templ);
                            }
                            recv_templ = SUPERLU_MALLOC(sizeof(double) * lda * nsupc);
                            recv_templ_len = lda * nsupc;
                        }
                        memcpy(recv_templ, valuel, sizeof(double) * lda * nsupc);
                    }
                }
            }
        }
        else
        {
            if (pcol == level % LYD_PGRID_Q)
            {
                for (int_t target_pcol = 0; target_pcol < LYD_PGRID_Q; target_pcol++)
                {
                    if (target_pcol != iam % LYD_PGRID_Q)
                    {
                        MPI_Request req;
                        MPI_Send(metal, 0, mpi_int_t, prow * LYD_PGRID_Q + target_pcol, 2, MPI_COMM_WORLD);
                        // MPI_Send(valuel, 0, MPI_DOUBLE, prow * LYD_PGRID_Q + target_pcol, 3, MPI_COMM_WORLD);
                    }
                    else
                    {
                        if (2 > recv_metal_len)
                        {
                            if (recv_metal)
                            {
                                SUPERLU_FREE(recv_metal);
                            }
                            // recv_metal = SUPERLU_MALLOC(sizeof(int_t) * 2);
                            // recv_metal_len = 2;
                            if (level % 2)
                            {
                                recv_metal_even = SUPERLU_MALLOC(sizeof(int_t) * 2);
                                recv_metal_len_even = 2;
                            }
                            else
                            {
                                recv_metal_odd = SUPERLU_MALLOC(sizeof(int_t) * 2);
                                recv_metal_len_odd = 2;
                            }
                        }
                        recv_metal[0] = 0;
                        recv_metal[1] = 0;
                    }
                }
            }
        }
        if (pcol != level % LYD_PGRID_Q)
        {
            MPI_Status mpi_stat;
            int count = 0;

            timer_start(&tv_start2);
            MPI_Probe(prow * LYD_PGRID_Q + level % LYD_PGRID_Q, 2, MPI_COMM_WORLD, &mpi_stat);
            time_tmp5 += timer_end(&tv_start2);
            MPI_Get_count(&mpi_stat, mpi_int_t, &count);
            if (count != 0)
            {
                if (count > recv_metal_len)
                {
                    if (recv_metal)
                    {
                        SUPERLU_FREE(recv_metal);
                    }
                    // recv_metal = SUPERLU_MALLOC(sizeof(int_t) * count);
                    // recv_metal_len = count;
                    if (level % 2)
                    {
                        recv_metal_even = SUPERLU_MALLOC(sizeof(int_t) * count);
                        recv_metal_len_even = count;
                    }
                    else
                    {
                        recv_metal_odd = SUPERLU_MALLOC(sizeof(int_t) * count);
                        recv_metal_len_odd = count;
                    }
                }
                MPI_Recv(recv_metal, count, mpi_int_t, prow * LYD_PGRID_Q + (level % LYD_PGRID_Q), 2, MPI_COMM_WORLD, &mpi_stat);
                lda = recv_metal[1];
                nrow_effective = recv_metal[1];

                int fetch_rank = prow * LYD_PGRID_Q + level % LYD_PGRID_Q;
                timer_start(&tv_start2);
                MPI_Probe(fetch_rank, 3, MPI_COMM_WORLD, &mpi_stat);
                time_tmp5 += timer_end(&tv_start2);
                MPI_Get_count(&mpi_stat, MPI_DOUBLE, &count);
                if (count > recv_templ_len)
                {
                    if (recv_templ)
                    {
                        SUPERLU_FREE(recv_templ);
                    }
                    recv_templ = SUPERLU_MALLOC(sizeof(double) * count);
                    recv_templ_len = count;
                }
                MPI_Recv(recv_templ, count, MPI_DOUBLE, fetch_rank, 3, MPI_COMM_WORLD, &mpi_stat);
                #ifdef USE_CUDA_AWARE_MPI
                if(level%2 == 1){ // odd
                    MPI_Recv(l_tile_offset_odd_d, (recv_metal[0]+1), mpi_int_t, fetch_rank, 101, MPI_COMM_WORLD, &mpi_stat);
                    MPI_Recv(l_bcol_localperm_odd_d, recv_metal[1], mpi_int_t, fetch_rank, 102, MPI_COMM_WORLD, &mpi_stat);
                    MPI_Recv(l_brow_idx_odd_d, recv_metal[0], mpi_int_t, fetch_rank, 103, MPI_COMM_WORLD, &mpi_stat);
                }else{ // even
                    MPI_Recv(l_tile_offset_even_d, (recv_metal[0]+1), mpi_int_t, fetch_rank, 101, MPI_COMM_WORLD, &mpi_stat);
                    MPI_Recv(l_bcol_localperm_even_d, recv_metal[1], mpi_int_t, fetch_rank, 102, MPI_COMM_WORLD, &mpi_stat);
                    MPI_Recv(l_brow_idx_even_d, recv_metal[0], mpi_int_t, fetch_rank, 103, MPI_COMM_WORLD, &mpi_stat);
                }
                #else
                if(level%2 == 1){ // odd
                    MPI_Recv(l_tile_offset_odd_h, (recv_metal[0]+1), mpi_int_t, fetch_rank, 101, MPI_COMM_WORLD, &mpi_stat);
                    cudaMemcpy(l_tile_offset_odd_d, l_tile_offset_odd_h, sizeof(mpi_int_t) * (recv_metal[0]+1), cudaMemcpyHostToDevice);
                    MPI_Recv(l_bcol_localperm_odd_h, recv_metal[1], mpi_int_t, fetch_rank, 102, MPI_COMM_WORLD, &mpi_stat);
                    cudaMemcpy(l_bcol_localperm_odd_d, l_bcol_localperm_odd_h, sizeof(mpi_int_t) * recv_metal[1], cudaMemcpyHostToDevice);
                    MPI_Recv(l_brow_idx_odd_h, recv_metal[0], mpi_int_t, fetch_rank, 103, MPI_COMM_WORLD, &mpi_stat);
                    cudaMemcpy(l_brow_idx_odd_d, l_brow_idx_odd_h, sizeof(mpi_int_t) * recv_metal[0], cudaMemcpyHostToDevice);
                }else{ // even
                    MPI_Recv(l_tile_offset_even_h, (recv_metal[0]+1), mpi_int_t, fetch_rank, 101, MPI_COMM_WORLD, &mpi_stat);
                    cudaMemcpy(l_tile_offset_even_d, l_tile_offset_even_h, sizeof(mpi_int_t) * (recv_metal[0]+1), cudaMemcpyHostToDevice);
                    MPI_Recv(l_bcol_localperm_even_h, recv_metal[1], mpi_int_t, fetch_rank, 102, MPI_COMM_WORLD, &mpi_stat);
                    cudaMemcpy(l_bcol_localperm_even_d, l_bcol_localperm_even_h, sizeof(mpi_int_t) * recv_metal[1], cudaMemcpyHostToDevice);
                    MPI_Recv(l_brow_idx_even_h, recv_metal[0], mpi_int_t, fetch_rank, 103, MPI_COMM_WORLD, &mpi_stat);
                    cudaMemcpy(l_brow_idx_even_d, l_brow_idx_even_h, sizeof(mpi_int_t) * recv_metal[0], cudaMemcpyHostToDevice);
                }
                #endif
            }
            else
            {
                if (2 > recv_metal_len)
                {
                    if (recv_metal)
                    {
                        SUPERLU_FREE(recv_metal);
                    }
                    // recv_metal = SUPERLU_MALLOC(sizeof(int_t) * 2);
                    // recv_metal_len = 2;
                    if (level % 2)
                    {
                        recv_metal_even = SUPERLU_MALLOC(sizeof(int_t) * 2);
                        recv_metal_len_even = 2;
                    }
                    else
                    {
                        recv_metal_odd = SUPERLU_MALLOC(sizeof(int_t) * 2);
                        recv_metal_len_odd = 2;
                    }
                }
                // timer_start(&tv_start2);
                MPI_Recv(recv_metal, 0, mpi_int_t, prow * LYD_PGRID_Q + (level % LYD_PGRID_Q), 2, MPI_COMM_WORLD, &mpi_stat);
                // time_tmp5 += timer_end(&tv_start2);
                recv_metal[0] = 0;
                recv_metal[1] = 0;
            }
        }
        time_tmp4 += timer_end(&tv_start1);

        // cudaDeviceSynchronize();
        // SYNC THREADPOOL
        if (flying_virtual_tid)
        {
            join_thread(flying_virtual_tid, trojan_horse_thread_pool);
        }


        if (!((!recv_metau) || (recv_metau[0] == 0) || (!recv_metal) || (recv_metal[0] == 0) || (nrow_effective == 0) || (ncol_effective == 0)))
        {
            // timer_start(&tv_start1);
            // MPI_Barrier(MPI_COMM_WORLD);
            // time_tmp1 += timer_end(&tv_start1);
            // printf("------#%d level=%d continue---------\n", iam, level);
            // continue;

            if (recv_metal)
            {
                expand_double_ptr_cuda(&bigcol_buf_d, &bigcol_buf_len, nsupc * nrow_effective);
                cudaMemcpyAsync(bigcol_buf_d, recv_templ, sizeof(double) * nsupc * nrow_effective, cudaMemcpyHostToDevice, stream);
            }
            if (recv_metau)
            {
                expand_double_ptr_cuda(&bigrow_buf_d, &bigrow_buf_len, ncol_effective * nsupc);
                cudaMemcpyAsync(bigrow_buf_d, recv_tempu, sizeof(double) * ncol_effective * nsupc, cudaMemcpyHostToDevice, stream);
            }

            // SSSSM update high_prio nsupers
            {
                int high_prio_nsupers = 1;

                int_t ll_nblk = recv_metal[0];
                int_t ll_lda = recv_metal[1];
                int_t lu_nblk = recv_metau[0];
                int_t lu_lda = SuperSize(level);
                int_t lu_metaidx = 3;
                int_t lu_blkidx = 0;
                double *lvalueu = recv_tempu;

                int_t uu_nblk = recv_metau[0];
                int_t uu_lda = SuperSize(level);
                int_t ul_nblk = recv_metal[0];
                int_t ul_lda = recv_metal[1];
                int_t ul_metaidx = 2;
                int_t ul_blkidx = 0;
                double *uvaluel = recv_templ;

                timer_start(&tv_start1);
                for (int_t bcr = level + 1; bcr < SUPERLU_MIN(nsupers, level + 1 + high_prio_nsupers); bcr++)
                {
                    if ((bcr % LYD_PGRID_Q) == (iam % LYD_PGRID_Q))
                    {
                        int_t *metaa = Lrowind_bc_ptr[LBj(bcr, grid)];
                        if (metaa)
                        {
                            while (lu_metaidx < recv_metau[2] && recv_metau[lu_metaidx + 0] < bcr)
                            {
                                lvalueu += recv_metau[lu_metaidx + 1];
                                lu_blkidx++;
                                lu_metaidx += 2 + SuperSize(recv_metau[lu_metaidx + 0]);
                            }
                            if ((lu_metaidx < recv_metau[2]) && (recv_metau[lu_metaidx + 0] == bcr))
                            {

                                double *valuel = recv_templ;
                                int_t l_metaidx = 2;
                                int_t l_blkidx = 0;
                                int_t l_lda = recv_metal[1];

                                double *valuea = Lnzval_bc_ptr[LBj(bcr, grid)];
                                int_t a_metaidx = 2;
                                int_t a_blkidx = 0;
                                int_t a_lda = metaa[1];
                                int_t a_nblk = metaa[0];
                                // int_t ncol_effective_local = get_ncol_effective(recv_metau+u_metaidx, level, xsup, SuperSize(recv_metau[u_metaidx+0]));
                                int_t ncol_effective_local = recv_metau[lu_metaidx + 1] / nsupc;
                                // printf("%d %d %d\n", recv_metau[lu_metaidx+1], nsupc, ncol_effective_local);

                                while (a_blkidx < a_nblk && l_blkidx < ll_nblk)
                                {
                                    if (metaa[a_metaidx + 0] < recv_metal[l_metaidx + 0])
                                    {
                                        valuea += metaa[a_metaidx + 1];
                                        a_blkidx++;
                                        a_metaidx += (2 + metaa[a_metaidx + 1]);
                                    }
                                    else if (metaa[a_metaidx + 0] > recv_metal[l_metaidx + 0])
                                    {
                                        valuel += recv_metal[l_metaidx + 1];
                                        l_blkidx++;
                                        l_metaidx += (2 + recv_metal[l_metaidx + 1]);
                                    }
                                    else
                                    {
                                        timer_start(&tv_start3);
                                        cblas_dgemm(
                                            CblasColMajor, CblasNoTrans, CblasNoTrans,
                                            recv_metal[l_metaidx + 1], ncol_effective_local, nsupc,
                                            1.0, valuel, l_lda, lvalueu, nsupc, 0.0, ssssm_result, recv_metal[l_metaidx + 1]);
                                        time_tstrf_update_kernel += timer_end(&tv_start3);

                                        int_t fnz = FstBlockC(metaa[a_metaidx + 0]);
                                        int_t dest_nbrow = metaa[a_metaidx + 1];
                                        int_t l_nbrow = recv_metal[l_metaidx + 1];
                                        int_t rel;
                                        int_t perm_i;

    #pragma omp simd
                                        for (perm_i = 0; perm_i < dest_nbrow; ++perm_i)
                                        {
                                            rel = metaa[a_metaidx + 2 + perm_i] - fnz;
                                            indirect_l_1[rel] = perm_i;
                                        }

    #pragma omp simd
                                        for (perm_i = 0; perm_i < l_nbrow; ++perm_i)
                                        { /* Source index is a subset of dest. */
                                            int rel = recv_metal[l_metaidx + 2 + perm_i] - fnz;
                                            indirect_l_2[perm_i] = indirect_l_1[rel];
                                        }

                                        int_t col = 0;
                                        int_t klst = FstBlockC(level + 1);
                                        int_t segsize;
                                        double *ssssm_result_local = ssssm_result;
                                        double *valuea_local = valuea;
                                        for (int col = 0; col < SuperSize(recv_metau[lu_metaidx + 0]); col++)
                                        {
                                            segsize = klst - recv_metau[lu_metaidx + 2 + col];
                                            if (segsize)
                                            {
    #pragma omp simd
                                                for (int rowidx = 0; rowidx < l_nbrow; rowidx++)
                                                {
    #pragma omp atomic
                                                    valuea_local[indirect_l_2[rowidx]] -= ssssm_result_local[rowidx];
                                                }
                                                ssssm_result_local += l_nbrow;
                                            }
                                            valuea_local += a_lda;
                                        }

                                        valuel += recv_metal[l_metaidx + 1];
                                        l_blkidx++;
                                        l_metaidx += (2 + recv_metal[l_metaidx + 1]);

                                        valuea += metaa[a_metaidx + 1];
                                        a_blkidx++;
                                        a_metaidx += (2 + metaa[a_metaidx + 1]);
                                    }
                                }
                            }
                        }
                    }

                    if ((bcr % LYD_PGRID_P) == (iam / LYD_PGRID_Q))
                    {
                        int_t *metaa = Ufstnz_br_ptr[LBi(bcr, grid)];
                        if (metaa)
                        {
                            int_t l_meta_len = (2 + recv_metal[0] * 2 + recv_metal[1]);
                            while ((ul_metaidx < l_meta_len) && (recv_metal[ul_metaidx + 0] < bcr))
                            {
                                ul_blkidx++;
                                uvaluel += recv_metal[ul_metaidx + 1];
                                ul_metaidx += 2 + recv_metal[ul_metaidx + 1];
                            }
                            if ((ul_metaidx < l_meta_len) && (recv_metal[ul_metaidx + 0] == bcr))
                            {
                                double *valuea = Unzval_br_ptr[LBi(bcr, grid)];
                                int_t u_metaidx = 3;
                                int_t u_blkidx = 0;
                                double *valueu = recv_tempu;
                                int_t a_metaidx = 3;
                                int_t a_blkidx = 0;
                                int_t a_nblk = metaa[0];
                                int_t a_lda = SuperSize(bcr);
                                int_t l_a_fstr = FstBlockC(recv_metal[ul_metaidx + 0]);
                                int_t l_nrow = recv_metal[ul_metaidx + 1];

                                // timer_start(&tv_start3);
                                // memset(l_fullrow_buf, 0, sizeof(double) * a_lda * nsupc);
                                // for (int_t l_col = 0; l_col < nsupc; l_col++)
                                // {
                                //     for(int_t rowidx = 0; rowidx < l_nrow; rowidx++){
                                //         int l_row = recv_metal[ul_metaidx + 2 + rowidx] - l_a_fstr;
                                //         l_fullrow_buf[l_col * a_lda + l_row] = uvaluel[l_col * ul_lda + rowidx];
                                //     }
                                // }
                                // time_tmp1 += timer_end(&tv_start3);

                                int_t a_fstbcol = metaa[3];
                                int_t ncol_skip = 0;
                                while ((u_blkidx < uu_nblk) && (recv_metau[u_metaidx + 0] < a_fstbcol))
                                {
                                    u_blkidx++;
                                    valueu += recv_metau[u_metaidx + 1];
                                    ncol_skip += (recv_metau[u_metaidx + 1] / nsupc);
                                    u_metaidx += (2 + SuperSize(recv_metau[u_metaidx + 0]));
                                }

                                int_t ncol_effective_brow = (recv_metau[1] / nsupc) - ncol_skip;
                                if (ssssm_result_cap < l_nrow * ncol_effective_brow)
                                {
                                    ssssm_result_cap = l_nrow * ncol_effective_brow;
                                    free(ssssm_result);
                                    ssssm_result = malloc(sizeof(double) * ssssm_result_cap);
                                }
                                timer_start(&tv_start3);
                                cblas_dgemm(
                                    CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    l_nrow, ncol_effective_brow, nsupc,
                                    1.0, uvaluel, ul_lda, valueu, nsupc, 0.0, ssssm_result, l_nrow);
                                time_gessm_update_kernel += timer_end(&tv_start3);

                                double *ssssm_result_ptr = ssssm_result;

                                for (int perm_i = 0; perm_i < l_nrow; ++perm_i)
                                { /* Source index is a subset of dest. */
                                    int rel = recv_metal[ul_metaidx + 2 + perm_i] - l_a_fstr;
                                    indirect_l_2[perm_i] = rel;
                                }

                                while (a_blkidx < a_nblk && u_blkidx < uu_nblk)
                                {
                                    while ((a_blkidx < a_nblk) && (metaa[a_metaidx + 0] < recv_metau[u_metaidx + 0]))
                                    {
                                        a_blkidx++;
                                        valuea += metaa[a_metaidx + 1];
                                        a_metaidx += (2 + SuperSize(metaa[a_metaidx + 0]));
                                    }
                                    while ((u_blkidx < uu_nblk) && (metaa[a_metaidx + 0] > recv_metau[u_metaidx + 0]))
                                    {
                                        u_blkidx++;
                                        ssssm_result_ptr += (recv_metau[u_metaidx + 1] / nsupc) * l_nrow;
                                        u_metaidx += (2 + SuperSize(recv_metau[u_metaidx + 0]));
                                    }
                                    if ((a_blkidx < a_nblk) && (u_blkidx < uu_nblk) && (metaa[a_metaidx + 0] == recv_metau[u_metaidx + 0]))
                                    {
                                        timer_start(&tv_start3);
                                        int_t segsize_a, segsize_u;
                                        int_t a_fnz = FstBlockC(bcr + 1), u_fnz = FstBlockC(level + 1);
                                        int_t ncol = SuperSize(recv_metau[u_metaidx + 0]);
                                        double *a_tile_buffer_local = valuea;
                                        double *ssssm_result_local = ssssm_result_ptr;
                                        for (int_t col = 0; col < ncol; col++)
                                        {
                                            segsize_a = a_fnz - metaa[a_metaidx + 2 + col];
                                            segsize_u = u_fnz - recv_metau[u_metaidx + 2 + col];
                                            if (segsize_a)
                                            {
                                                if (segsize_u)
                                                {
    // #pragma omp simd
    // for (int_t row = 0; row < a_lda; row++)
    // {
    //     a_tile_buffer_local[row] -= ssssm_result_local[row];
    // }
    #pragma omp simd
                                                    for (int_t rowidx = 0; rowidx < l_nrow; rowidx++)
                                                    {
    // printf("rowidx=%d indirectl2=%d %p\n", rowidx, indirect_l_2[rowidx], a_tile_buffer_local + indirect_l_2[rowidx]);
    #pragma omp atomic
                                                        a_tile_buffer_local[indirect_l_2[rowidx]] -= ssssm_result_local[rowidx];
                                                    }
                                                    ssssm_result_local += l_nrow;
                                                }
                                                a_tile_buffer_local += a_lda;
                                            }
                                            else if (segsize_u)
                                            {
                                                ssssm_result_local += l_nrow;
                                            }
                                        }
                                        time_tmp3 += timer_end(&tv_start3);

                                        u_blkidx++;
                                        ssssm_result_ptr += (recv_metau[u_metaidx + 1] / nsupc) * l_nrow;
                                        u_metaidx += (2 + SuperSize(recv_metau[u_metaidx + 0]));

                                        a_blkidx++;
                                        valuea += metaa[a_metaidx + 1];
                                        a_metaidx += (2 + SuperSize(metaa[a_metaidx + 0]));
                                    }
                                }
                            }
                        }
                    }
                }
                time_tmp1 += timer_end(&tv_start1);
            }
        }


        // START THREADPOOL FUNC
        thread_param->level = level;
        thread_param->ncol_effective = ncol_effective;
        thread_param->nrow_effective = nrow_effective;
        thread_param->nsupc = nsupc;

        if(pcol == level % LYD_PGRID_Q){ // L (local)
            thread_param->l_tile_offset_d = Llu->l_tile_offset_hd[level];
            thread_param->l_bcol_localperm_d = Llu->l_bcol_localperm_hd[level];
            thread_param->l_brow_idx_d = Llu->l_brow_idx_hd[level];
        }else if((level % 2) == 1){ // odd (remote)
            thread_param->l_tile_offset_d = l_tile_offset_odd_d;
            thread_param->l_bcol_localperm_d = l_bcol_localperm_odd_d;
            thread_param->l_brow_idx_d = l_brow_idx_odd_d;
        }else{
            thread_param->l_tile_offset_d = l_tile_offset_even_d;
            thread_param->l_bcol_localperm_d = l_bcol_localperm_even_d;
            thread_param->l_brow_idx_d = l_brow_idx_even_d;
        }

        if(prow == level % LYD_PGRID_P){ // U (local)
            thread_param->u_tile_offset_d = Llu->u_tile_offset_hd[level];
            thread_param->u_brow_localperm_d = Llu->u_brow_localperm_hd[level];
            thread_param->u_bcol_idx_d = Llu->u_bcol_idx_hd[level];
        }else if((level % 2) == 1){ // odd (remote)
            thread_param->u_tile_offset_d = u_tile_offset_odd_d;
            thread_param->u_brow_localperm_d = u_brow_localperm_odd_d;
            thread_param->u_bcol_idx_d = u_bcol_idx_odd_d;
        }else{
            thread_param->u_tile_offset_d = u_tile_offset_even_d;
            thread_param->u_brow_localperm_d = u_brow_localperm_even_d;
            thread_param->u_bcol_idx_d = u_bcol_idx_even_d;
        }
        flying_virtual_tid = alloc_thread(trojan_horse_thread_pool, trojan_horse_batched_kernel, thread_param, 1);

        // cudaDeviceSynchronize();
        // time_tstrf_update += timer_end(&tv_start1);

        // MPI_Barrier(MPI_COMM_WORLD);
        // printf("------#%d level=%d done---------\n", iam, level);
    }

    // if (iam == 0)
    // {
    //     printf("rank\tinit\tgetrf\tgetrfK\tgetrfS\tgetrfR\tgetrfU\tgetrfUK\ttstrf\ttstrfK\ttstrfS\ttstrfR\ttstrfU\ttstrfUK\tgessm\tgessmK\tgessmS\tgessmR\tgessmU\tgessmUK\tdivU\tdivL\n");
    //     fflush(stdout);
    // }
    // for (int i = 0; i < nproc; i++)
    // {
    //     if (iam == i)
    //     {
    //         printf("%d\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\n",
    //                iam,
    //                time_init,
    //                time_getrf_all, time_getrf_kernel, time_getrf_send, time_getrf_recv, time_getrf_update, time_getrf_update_kernel,
    //                time_tstrf_all, time_tstrf_kernel, time_tstrf_send, time_tstrf_recv, time_tstrf_update, time_tstrf_update_kernel,
    //                time_gessm_all, time_gessm_kernel, time_gessm_send, time_gessm_recv, time_gessm_update, time_gessm_update_kernel,
    //                time_divide_u, time_divide_l);
    //         printf("%d\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\n",
    //                iam, time_tmp1, time_tmp2, time_tmp3, time_tmp4, time_tmp5, time_tmp6, time_tmp7);
    //         fflush(stdout);
    //     }
    //     usleep(100);
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    stat->time_tstrf_update = time_tstrf_update;

    return 0;
}
