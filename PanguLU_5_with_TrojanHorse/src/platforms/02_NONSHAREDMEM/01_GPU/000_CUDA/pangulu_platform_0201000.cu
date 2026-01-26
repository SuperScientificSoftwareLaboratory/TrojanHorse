#define PANGULU_PLATFORM_ENV
#include "../../../../pangulu_common.h"

// Need to remove
#define TROJAN_HORSE_SSSSM_BATCHED_THREAD_PER_ELEM 8

#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
#define CHECK_CUDA_LAST_ERROR() check_cuda_last_error(__FILE__, __LINE__)
#define PANGULU_WARP_SIZE 32
#define PANGULU_THREAD_PER_BLOCK (pangulu_gpu_kernel_warp_per_block * PANGULU_WARP_SIZE)
#define PANGULU_DATAMOV_THREADPERBLOCK (pangulu_gpu_data_move_warp_per_block * PANGULU_WARP_SIZE)
extern int pangulu_gpu_kernel_warp_per_block;
extern int pangulu_gpu_data_move_warp_per_block;
extern int pangulu_gpu_shared_mem_size;

#include <cublas_v2.h>
#include <cusolverDn.h>
cublasHandle_t cublas_handle;
cusolverDnHandle_t cusolver_handle;

extern char mtx_name_glo[100];
calculate_type *getrf_dense_buf_d;
calculate_type *d_cusolver_work;
int *d_cusolver_info;
int *d_cusolver_pivot;
calculate_type *d_getrf_tag_buffer = NULL;
calculate_type *d_ssssm_dense_buf_opdst = NULL;
calculate_type *d_ssssm_dense_buf_op1 = NULL;
calculate_type *d_ssssm_dense_buf_op2 = NULL;

pangulu_uint64_t task_pointer_buf_capacity = 0;

pangulu_inblock_ptr **hd_rowptrc;
pangulu_inblock_idx **hd_colidxc;
calculate_type **hd_valuec;
pangulu_inblock_ptr **hd_rowptrb;
pangulu_inblock_idx **hd_colidxb;
calculate_type **hd_valueb;
pangulu_inblock_ptr **hd_rowptra;
pangulu_inblock_idx **hd_colidxa;
calculate_type **hd_valuea;

pangulu_inblock_ptr **dd_rowptrc;
pangulu_inblock_idx **dd_colidxc;
calculate_type **dd_valuec;
pangulu_inblock_ptr **dd_rowptrb;
pangulu_inblock_idx **dd_colidxb;
calculate_type **dd_valueb;
pangulu_inblock_ptr **dd_rowptra;
pangulu_inblock_idx **dd_colidxa;
calculate_type **dd_valuea;

pangulu_int32_t *h_task_types;
pangulu_int32_t *d_task_types;

pangulu_int32_t *h_task_block_ptr;
pangulu_int32_t *d_task_block_ptr;

calculate_type **hd_getrf_tag_double;
pangulu_int32_t **hd_getrf_nnzu;
pangulu_inblock_ptr **hd_getrf_csccolptrl_upperbound;
pangulu_inblock_idx **hd_getrf_cscrowidxl_upperbound;
pangulu_inblock_ptr **hd_getrf_csccolptru_upperbound;
pangulu_inblock_idx **hd_getrf_cscrowidxu_upperbound;
pangulu_inblock_ptr **hd_tstrf_a_valueidx;
pangulu_inblock_ptr **hd_tstrf_l_valueidx;

calculate_type **dd_getrf_tag_double;
pangulu_int32_t **dd_getrf_nnzu;
pangulu_inblock_ptr **dd_getrf_csccolptrl_upperbound;
pangulu_inblock_idx **dd_getrf_cscrowidxl_upperbound;
pangulu_inblock_ptr **dd_getrf_csccolptru_upperbound;
pangulu_inblock_idx **dd_getrf_cscrowidxu_upperbound;
pangulu_inblock_ptr **dd_tstrf_a_valueidx;
pangulu_inblock_ptr **dd_tstrf_l_valueidx;

pangulu_inblock_ptr *d_general_dense_columnpointer = NULL;
pangulu_inblock_idx *d_general_dense_rowindex = NULL;
calculate_type *d_dense_buffer = NULL;
pangulu_uint64_t dense_buffer_block_cap = 0;
pangulu_uint64_t *dense_task_indeces = NULL;
pangulu_uint64_t dense_task_indeces_cap = 0;
calculate_type **hd_ssssm_batch_ptr;
calculate_type **dd_ssssm_batch_ptr;

pangulu_uint64_t getrf_buffer_cap = 0;
pangulu_uint64_t *getrf_indeces = NULL;
pangulu_uint64_t getrf_indeces_cap = 0;

char *info_pool_h = NULL;
char *info_pool_d = NULL;

#ifdef PANGULU_PERF
extern pangulu_stat_t global_stat;
#endif

void check_cuda_last_error(const char *const file, int const line)
{
    cudaError_t result = cudaGetLastError();
    if (result)
    {
        fprintf(stderr, "[PanguLU ERROR] CUDA error at %s:%d %s (code=%d)\n",
                file, line, cudaGetErrorString(result), static_cast<unsigned int>(result));
        exit(EXIT_FAILURE);
    }
}

void check_cuda_error(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "[PanguLU ERROR] CUDA error at %s:%d %s (code=%d)\n",
                file, line, cudaGetErrorString(result), static_cast<unsigned int>(result));
        exit(EXIT_FAILURE);
    }
}

void pangulu_platform_0201000_malloc(void **platform_address, size_t size)
{
    cudaError_t err = cudaMalloc(platform_address, size);
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_malloc_pinned(void **platform_address, size_t size)
{
    cudaError_t err = cudaHostAlloc(platform_address, size, cudaHostAllocDefault);
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_synchronize()
{
    cudaError_t err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_memset(void *s, int c, size_t n)
{
    cudaError_t err = cudaMemset(s, c, n);
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_create_stream(void **stream)
{
    cudaError_t err = cudaStreamCreate((cudaStream_t *)stream);
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_memcpy(void *dst, const void *src, size_t count, unsigned int kind)
{
    cudaError_t err;
    if (kind == 0)
    {
        err = cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
    }
    else if (kind == 1)
    {
        err = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
    }
    else if (kind == 2)
    {
        err = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
    }
    else
    {
        fprintf(stderr, "Invalid memcpy kind value\n");
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_memcpy_async(void *dst, const void *src, size_t count, unsigned int kind, void *stream)
{
    cudaError_t err;
    if (kind == 0)
    {
        err = cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, (cudaStream_t)stream);
    }
    else if (kind == 1)
    {
        err = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
    }
    else if (kind == 2)
    {
        err = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    }
    else
    {
        fprintf(stderr, "Invalid memcpy kind value\n");
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_free(void *devptr)
{
    cudaError_t err = cudaFree(devptr);
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_get_device_num(int *device_num)
{
    cudaError_t err = cudaGetDeviceCount(device_num);
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_set_default_device(int device_num)
{
    cudaError_t err = cudaSetDevice(device_num);
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_num);
    CHECK_CUDA_ERROR(err);
    pangulu_gpu_shared_mem_size = prop.sharedMemPerBlock;
}

void pangulu_platform_0201000_get_device_name(char *name, int device_num)
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_num);
    CHECK_CUDA_ERROR(err);
    strcpy(name, prop.name);
}

void pangulu_platform_0201000_get_device_memory_usage(size_t *used_byte)
{
    size_t total_byte;
    cudaError_t err = cudaMemGetInfo(used_byte, &total_byte);
    *used_byte = total_byte - *used_byte;
    CHECK_CUDA_ERROR(err);
}

#ifdef PANGULU_NONSHAREDMEM

void pangulu_cuda_download_block(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *slot)
{
    pangulu_platform_0201000_memcpy(slot->value, slot->d_value, sizeof(calculate_type) * slot->columnpointer[nb], 1);
}

__global__ void pangulu_load_dense(
    int nb,
    pangulu_inblock_ptr *columnpointer,
    pangulu_inblock_idx *rowindex,
    calculate_real_type *value_rp,
    calculate_real_type *dense_buf_rp)
{
    calculate_type *value = (calculate_type *)value_rp;
    calculate_type *dense_buf = (calculate_type *)dense_buf_rp;
    int col = blockIdx.x;
    if (col >= nb)
    {
        return;
    }
    for (pangulu_inblock_idx row = threadIdx.x; row < nb; row += blockDim.x)
    {
#ifdef PANGULU_COMPLEX
        __real__(dense_buf[col * nb + row]) = 0;
        __imag__(dense_buf[col * nb + row]) = 0;
#else
        dense_buf[col * nb + row] = 0;
#endif
    }
    __syncthreads();
    for (int idx = columnpointer[col] + threadIdx.x; idx < columnpointer[col + 1]; idx += blockDim.x)
    {
        int row = rowindex[idx];
        dense_buf[col * nb + row] = value[idx];
    }
}

__global__ void pangulu_load_dense_getrf(
    int nb,
    pangulu_inblock_ptr *columnpointer,
    pangulu_inblock_idx *rowindex,
    calculate_real_type *value_csc_rp,
    pangulu_inblock_ptr *rowpointer,
    pangulu_inblock_idx *columnindex,
    calculate_real_type *value_csr_rp,
    calculate_real_type *dense_buf_rp)
{
    calculate_type *value_csc = (calculate_type *)value_csc_rp;
    calculate_type *value_csr = (calculate_type *)value_csr_rp;
    calculate_type *dense_buf = (calculate_type *)dense_buf_rp;

    int rc = blockIdx.x;
    if (rc >= nb)
    {
        return;
    }
    // for(pangulu_inblock_idx row = threadIdx.x; row < nb; row+=blockDim.x){
    //     dense_buf[rc * nb + row] = 0;
    // }
    // __syncthreads();
    for (int idx = columnpointer[rc] + threadIdx.x; idx < columnpointer[rc + 1]; idx += blockDim.x)
    {
        int row = rowindex[idx];
        dense_buf[rc * nb + row] = value_csc[idx];
    }
    for (int idx = rowpointer[rc] + threadIdx.x; idx < rowpointer[rc + 1]; idx += blockDim.x)
    {
        int col = columnindex[idx];
        dense_buf[col * nb + rc] = value_csr[idx];
    }
}

__global__ void pangulu_store_dense_getrf(
    int nb,
    pangulu_inblock_ptr *columnpointer,
    pangulu_inblock_idx *rowindex,
    calculate_real_type *value_csc_rp,
    pangulu_inblock_ptr *rowpointer,
    pangulu_inblock_idx *columnindex,
    calculate_real_type *value_csr_rp,
    calculate_real_type *dense_buf_rp)
{
    calculate_type *value_csc = (calculate_type *)value_csc_rp;
    calculate_type *value_csr = (calculate_type *)value_csr_rp;
    calculate_type *dense_buf = (calculate_type *)dense_buf_rp;

    int rc = blockIdx.x;
    if (rc >= nb)
    {
        return;
    }
    for (int idx = columnpointer[rc] + threadIdx.x; idx < columnpointer[rc + 1]; idx += blockDim.x)
    {
        int row = rowindex[idx];
        value_csc[idx] = dense_buf[rc * nb + row];
    }
    for (int idx = rowpointer[rc] + threadIdx.x; idx < rowpointer[rc + 1]; idx += blockDim.x)
    {
        int col = columnindex[idx];
        value_csr[idx] = dense_buf[col * nb + rc];
    }
}

__global__ void pangulu_store_dense(
    int nb,
    pangulu_inblock_ptr *columnpointer,
    pangulu_inblock_idx *rowindex,
    calculate_type *value,
    calculate_type *dense_buf)
{
    int col = blockIdx.x;
    if (col >= nb)
    {
        return;
    }
    for (int idx = columnpointer[col] + threadIdx.x; idx < columnpointer[col + 1]; idx += blockDim.x)
    {
        int row = rowindex[idx];
        value[idx] = dense_buf[col * nb + row];
    }
}

__global__ void clear_dense(
    pangulu_inblock_idx nb,
    calculate_real_type *dense_rp)
{
    calculate_type *dense = (calculate_type *)dense_rp;
    int col = blockIdx.x;
    if (col >= nb)
    {
        return;
    }
    for (int idx = col * nb + threadIdx.x; idx < (col + 1) * nb; idx += blockDim.x)
    {
#ifdef PANGULU_COMPLEX
        __real__(dense[idx]) = 0.0;
        __imag__(dense[idx]) = 0.0;
#else
        dense[idx] = 0.0;
#endif
    }
}

__global__ void tstrf_cuda(
    pangulu_inblock_idx n,
    pangulu_inblock_ptr *b_rowpointer,
    pangulu_inblock_idx *b_columnindex,
    pangulu_inblock_ptr *b_valueidx,
    calculate_real_type *b_value_rp,
    pangulu_inblock_ptr *l_rowpointer,
    pangulu_inblock_idx *l_columnindex,
    calculate_real_type *l_value_rp)
{
    extern __shared__ char shared_memory[];
    pangulu_inblock_idx *s_idxa = (pangulu_inblock_idx *)shared_memory;
    calculate_type *s_dense = (calculate_type *)(shared_memory + sizeof(pangulu_inblock_idx) * n * (blockDim.x / PANGULU_WARP_SIZE));

    calculate_type *b_value = (calculate_type *)b_value_rp;
    calculate_type *l_value = (calculate_type *)l_value_rp;

    pangulu_inblock_idx colidx = blockIdx.x * (blockDim.x / PANGULU_WARP_SIZE) + (threadIdx.x / PANGULU_WARP_SIZE);
    if (colidx >= n)
    {
        return;
    }
    pangulu_inblock_idx warp_thread = threadIdx.x % PANGULU_WARP_SIZE;
    pangulu_inblock_idx *s_idxa_warp = s_idxa + (threadIdx.x / PANGULU_WARP_SIZE) * n;
    calculate_type *s_dense_warp = s_dense + (threadIdx.x / PANGULU_WARP_SIZE) * n;

    pangulu_inblock_ptr b_col_start = b_rowpointer[colidx];
    pangulu_inblock_ptr b_col_end = b_rowpointer[colidx + 1];
    if (b_col_end == b_col_start)
    {
        return;
    }

    for (pangulu_inblock_idx i = warp_thread; i < b_col_end - b_col_start; i += PANGULU_WARP_SIZE)
    {
        s_idxa_warp[i] = b_columnindex[b_col_start + i];
        s_dense_warp[s_idxa_warp[i]] = b_value[b_valueidx[b_col_start + i]];
    }

    for (pangulu_inblock_ptr i = b_col_start, t = 0; i < b_col_end; i++, t++)
    {
        pangulu_inblock_idx rowa = s_idxa_warp[t];
        pangulu_inblock_ptr coll1 = l_rowpointer[rowa];
        pangulu_inblock_ptr coll2 = l_rowpointer[rowa + 1];

        calculate_type vala;
        vala = s_dense_warp[s_idxa_warp[t]];
#ifdef PANGULU_COMPLEX
        calculate_type z1 = vala;
        calculate_type z2 = l_value[coll1];
        __real__(vala) =
            (__real__(z1) * __real__(z2) + __imag__(z1) * __imag__(z2)) /
            (__real__(z2) * __real__(z2) + __imag__(z2) * __imag__(z2));
        __imag__(vala) =
            (__imag__(z1) * __real__(z2) - __real__(z1) * __imag__(z2)) /
            (__real__(z2) * __real__(z2) + __imag__(z2) * __imag__(z2));
#else
        vala /= l_value[coll1];
#endif
        if (warp_thread == 0)
        {
            s_dense_warp[s_idxa_warp[t]] = vala;
        }

        for (pangulu_inblock_ptr j = coll1 + 1 + warp_thread, p = warp_thread; j < coll2; j += PANGULU_WARP_SIZE, p += PANGULU_WARP_SIZE)
        {
#ifdef PANGULU_COMPLEX
            z1 = vala;
            z2 = l_value[j];
            __real__(s_dense_warp[l_columnindex[j]]) -= (__real__(z1) * __real__(z2) - __imag__(z1) * __imag__(z2));
            __imag__(s_dense_warp[l_columnindex[j]]) -= (__imag__(z1) * __real__(z2) + __real__(z1) * __imag__(z2));
#else
            s_dense_warp[l_columnindex[j]] -= vala * l_value[j];
#endif
        }
    }

    for (pangulu_inblock_idx i = warp_thread; i < b_col_end - b_col_start; i += PANGULU_WARP_SIZE)
    {
        b_value[b_valueidx[b_col_start + i]] = s_dense_warp[s_idxa_warp[i]];
    }
}

__global__ void gessm_cuda(
    pangulu_inblock_idx n,
    pangulu_inblock_ptr *b_columnpointer,
    pangulu_inblock_idx *b_rowindex,
    calculate_real_type *b_value_rp,
    pangulu_inblock_ptr *l_columnpointer,
    pangulu_inblock_idx *l_rowindex,
    calculate_real_type *l_value_rp)
{
    extern __shared__ char shared_memory[];
    pangulu_inblock_idx *s_idxa = (pangulu_inblock_idx *)shared_memory;
    calculate_type *s_dense = (calculate_type *)(shared_memory + sizeof(pangulu_inblock_idx) * n * (blockDim.x / PANGULU_WARP_SIZE));

    calculate_type *b_value = (calculate_type *)b_value_rp;
    calculate_type *l_value = (calculate_type *)l_value_rp;

    pangulu_inblock_idx colidx = blockIdx.x * (blockDim.x / PANGULU_WARP_SIZE) + (threadIdx.x / PANGULU_WARP_SIZE);
    if (colidx >= n)
    {
        return;
    }
    pangulu_inblock_idx warp_thread = threadIdx.x % PANGULU_WARP_SIZE;
    pangulu_inblock_idx *s_idxa_warp = s_idxa + (threadIdx.x / PANGULU_WARP_SIZE) * n;
    calculate_type *s_dense_warp = s_dense + (threadIdx.x / PANGULU_WARP_SIZE) * n;

    pangulu_inblock_ptr b_col_start = b_columnpointer[colidx];
    pangulu_inblock_ptr b_col_end = b_columnpointer[colidx + 1];
    if (b_col_end == b_col_start)
    {
        return;
    }

    for (pangulu_inblock_idx i = warp_thread; i < b_col_end - b_col_start; i += PANGULU_WARP_SIZE)
    {
        s_idxa_warp[i] = b_rowindex[b_col_start + i];
        s_dense_warp[s_idxa_warp[i]] = b_value[b_col_start + i];
    }

    for (pangulu_inblock_ptr i = b_col_start, t = 0; i < b_col_end; i++, t++)
    {
        pangulu_inblock_idx rowa = s_idxa_warp[t];
        calculate_type vala = s_dense_warp[s_idxa_warp[t]];
        pangulu_inblock_ptr coll1 = l_columnpointer[rowa];
        pangulu_inblock_ptr coll2 = l_columnpointer[rowa + 1];
        for (pangulu_int64_t j = coll1 + warp_thread, p = warp_thread; j < coll2; j += PANGULU_WARP_SIZE, p += PANGULU_WARP_SIZE)
        {
#ifdef PANGULU_COMPLEX
            calculate_type z1 = vala;
            calculate_type z2 = l_value[j];
            __real__(s_dense_warp[l_rowindex[j]]) -= (__real__(z1) * __real__(z2) - __imag__(z1) * __imag__(z2));
            __imag__(s_dense_warp[l_rowindex[j]]) -= (__imag__(z1) * __real__(z2) + __real__(z1) * __imag__(z2));
#else
            s_dense_warp[l_rowindex[j]] -= vala * l_value[j];
#endif
        }
    }

    for (pangulu_inblock_idx i = warp_thread; i < b_col_end - b_col_start; i += PANGULU_WARP_SIZE)
    {
        b_value[b_col_start + i] = s_dense_warp[s_idxa_warp[i]];
    }
}

__global__ void ssssm_cuda(
    pangulu_inblock_idx n,
    pangulu_inblock_ptr *d_rowptrc,
    pangulu_inblock_idx *d_colidxc,
    calculate_real_type *d_valuec_rp,
    pangulu_inblock_ptr *d_rowptrb,
    pangulu_inblock_idx *d_colidxb,
    calculate_real_type *d_valueb_rp,
    pangulu_inblock_ptr *d_rowptra,
    pangulu_inblock_idx *d_colidxa,
    calculate_real_type *d_valuea_rp)
{
    extern __shared__ calculate_type s_dense[];
    pangulu_inblock_idx row = blockIdx.x;
    const pangulu_inblock_idx thread_offset = threadIdx.x;

    calculate_type *d_valuec = (calculate_type *)d_valuec_rp;
    calculate_type *d_valueb = (calculate_type *)d_valueb_rp;
    calculate_type *d_valuea = (calculate_type *)d_valuea_rp;

    if (row >= n)
    {
        return;
    }

    pangulu_inblock_ptr therowc = d_rowptrc[row];
    pangulu_inblock_ptr nextrowc = d_rowptrc[row + 1];

    pangulu_inblock_ptr therow = d_rowptra[row];
    pangulu_inblock_ptr nextrow = d_rowptra[row + 1];

    for (pangulu_inblock_ptr idx = therowc + thread_offset; idx < nextrowc; idx += blockDim.x)
    {
        pangulu_inblock_idx col = d_colidxc[idx];
#ifdef PANGULU_COMPLEX
        __real__(s_dense[col]) = 0.0;
        __imag__(s_dense[col]) = 0.0;
#else
        s_dense[col] = 0.0;
#endif
    }

    __syncthreads();

    for (pangulu_inblock_ptr i = therow; i < nextrow; i++)
    {
        pangulu_inblock_idx cola = d_colidxa[i];
        calculate_type vala = d_valuea[i];

        pangulu_inblock_ptr therowb = d_rowptrb[cola];
        pangulu_inblock_ptr nextrowb = d_rowptrb[cola + 1];

        for (pangulu_inblock_ptr j = therowb + thread_offset; j < nextrowb; j += blockDim.x)
        {
            pangulu_inblock_idx colb = d_colidxb[j];
#ifdef PANGULU_COMPLEX
            calculate_type z1 = vala;
            calculate_type z2 = d_valueb[j];
            atomicAdd(&__real__(s_dense[colb]), (__real__(z1) * __real__(z2) - __imag__(z1) * __imag__(z2)));
            atomicAdd(&__imag__(s_dense[colb]), (__imag__(z1) * __real__(z2) + __real__(z1) * __imag__(z2)));
#else
            atomicAdd(&s_dense[colb], vala * d_valueb[j]);
#endif
        }
    }

    __syncthreads();

    for (pangulu_inblock_ptr idx = therowc + thread_offset; idx < nextrowc; idx += blockDim.x)
    {
        pangulu_inblock_idx col = d_colidxc[idx];
#ifdef PANGULU_COMPLEX
        atomicAdd(&__real__(d_valuec[idx]), -__real__(s_dense[col]));
        atomicAdd(&__imag__(d_valuec[idx]), -__imag__(s_dense[col]));
#else
        atomicAdd(&d_valuec[idx], -s_dense[col]);
#endif
    }
}

void pangulu_platform_0201000_getrf(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    int tid)
{
    if (!d_getrf_tag_buffer)
    {
        cudaMalloc(&d_getrf_tag_buffer, sizeof(calculate_type) * nb * nb);
        if (!d_getrf_tag_buffer)
        {
            printf("[PanguLU Error] cudaMalloc failed: requested %lld bytes but returned NULL.\n", sizeof(calculate_type) * nb * nb);
        }
    }

    if (!cusolver_handle)
    {
        cusolverDnCreate(&cusolver_handle);
        int work_size = 0;
#if defined(CALCULATE_TYPE_R64)
        cusolverDnDgetrf_bufferSize(cusolver_handle, nb, nb, d_getrf_tag_buffer, nb, &work_size);
#elif defined(CALCULATE_TYPE_R32)
        cusolverDnSgetrf_bufferSize(cusolver_handle, nb, nb, d_getrf_tag_buffer, nb, &work_size);
#elif defined(CALCULATE_TYPE_CR64)
        cusolverDnZgetrf_bufferSize(cusolver_handle, nb, nb, (cuDoubleComplex *)d_getrf_tag_buffer, nb, &work_size);
#elif defined(CALCULATE_TYPE_CR32)
        cusolverDnCgetrf_bufferSize(cusolver_handle, nb, nb, (cuFloatComplex *)d_getrf_tag_buffer, nb, &work_size);
#else
#error [PanguLU Compile Error] Unsupported CALCULATE_TYPE for the selected BLAS library. Please recompile with a compatible value.
#endif
        cudaMalloc(&d_cusolver_work, work_size * sizeof(calculate_type));
        cudaMalloc(&d_cusolver_info, sizeof(int));
        cudaMalloc(&d_cusolver_pivot, sizeof(int) * nb);
    }

    pangulu_storage_slot_t *diag_upper = NULL;
    pangulu_storage_slot_t *diag_lower = NULL;
    if (opdst->is_upper)
    {
        diag_upper = opdst;
        diag_lower = opdst->related_block;
    }
    else
    {
        diag_upper = opdst->related_block;
        diag_lower = opdst;
    }

    clear_dense<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(nb, (calculate_real_type *)d_getrf_tag_buffer);
    cudaDeviceSynchronize();
    pangulu_load_dense_getrf<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(
        nb,
        diag_lower->d_columnpointer,
        diag_lower->d_rowindex,
        (calculate_real_type *)diag_lower->d_value,
        diag_upper->d_rowpointer,
        diag_upper->d_columnindex,
        (calculate_real_type *)diag_upper->d_value,
        (calculate_real_type *)d_getrf_tag_buffer);

#ifdef PANGULU_PERF
    struct timeval start;
    cudaDeviceSynchronize();
    pangulu_time_start(&start);
#endif

#if defined(CALCULATE_TYPE_R64)
    cusolverDnDgetrf(cusolver_handle, nb, nb, d_getrf_tag_buffer, nb, d_cusolver_work, NULL, d_cusolver_info);
#elif defined(CALCULATE_TYPE_R32)
    cusolverDnSgetrf(cusolver_handle, nb, nb, d_getrf_tag_buffer, nb, d_cusolver_work, NULL, d_cusolver_info);
#elif defined(CALCULATE_TYPE_CR64)
    cusolverDnZgetrf(cusolver_handle, nb, nb, (cuDoubleComplex *)d_getrf_tag_buffer, nb, (cuDoubleComplex *)d_cusolver_work, NULL, d_cusolver_info);
#elif defined(CALCULATE_TYPE_CR32)
    cusolverDnCgetrf(cusolver_handle, nb, nb, (cuFloatComplex *)d_getrf_tag_buffer, nb, (cuFloatComplex *)d_cusolver_work, NULL, d_cusolver_info);
#else
#error [PanguLU Compile Error] Unsupported CALCULATE_TYPE for the selected BLAS library. Please recompile with a compatible value.
#endif

#ifdef PANGULU_PERF
    cudaDeviceSynchronize();
    global_stat.time_inner_kernel += pangulu_time_stop(&start);
#endif

    pangulu_store_dense_getrf<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(
        nb,
        diag_lower->d_columnpointer,
        diag_lower->d_rowindex,
        (calculate_real_type *)diag_lower->d_value,
        diag_upper->d_rowpointer,
        diag_upper->d_columnindex,
        (calculate_real_type *)diag_upper->d_value,
        (calculate_real_type *)d_getrf_tag_buffer);

    pangulu_cuda_download_block(nb, diag_upper);
    pangulu_cuda_download_block(nb, diag_lower);
}

void pangulu_platform_0201000_tstrf(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *opdiag,
    int tid)
{
#ifdef PANGULU_PERF
    global_stat.kernel_cnt++;
    struct timeval start;
    pangulu_platform_0201000_synchronize();
    pangulu_time_start(&start);
#endif

    if (opdiag->is_upper == 0)
    {
        opdiag = opdiag->related_block;
    }

    int shared_memory_size = (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * nb * pangulu_gpu_kernel_warp_per_block;
    if (shared_memory_size > pangulu_gpu_shared_mem_size)
    {
        printf("[PanguLU Error] Requested shared memory size %d bytes exceeds the maximum limit of %d bytes.\n", shared_memory_size, pangulu_gpu_shared_mem_size);
        printf("[PanguLU Error] Please reduce 'init_options.nb' and try again. Exiting.\n");
        exit(1);
    }
    tstrf_cuda<<<
        PANGULU_ICEIL(nb, pangulu_gpu_kernel_warp_per_block),
        pangulu_gpu_kernel_warp_per_block * PANGULU_WARP_SIZE,
        shared_memory_size>>>(
        nb,
        opdst->d_rowpointer, opdst->d_columnindex, opdst->d_idx_of_csc_value_for_csr, (calculate_real_type *)opdst->d_value,
        opdiag->d_rowpointer, opdiag->d_columnindex, (calculate_real_type *)opdiag->d_value);

#ifdef PANGULU_PERF
    pangulu_platform_0201000_synchronize();
    global_stat.time_inner_kernel += pangulu_time_stop(&start);
#endif
    pangulu_cuda_download_block(nb, opdst);
}

void pangulu_platform_0201000_gessm(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *opdiag,
    int tid)
{
#ifdef PANGULU_PERF
    global_stat.kernel_cnt++;
    struct timeval start;
    pangulu_platform_0201000_synchronize();
    pangulu_time_start(&start);
#endif

    if (opdiag->is_upper == 1)
    {
        opdiag = opdiag->related_block;
    }

    int shared_memory_size = (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * nb * pangulu_gpu_kernel_warp_per_block;
    if (shared_memory_size > pangulu_gpu_shared_mem_size)
    {
        printf("[PanguLU Error] Requested shared memory size %d bytes exceeds the maximum limit of %d bytes.\n", shared_memory_size, pangulu_gpu_shared_mem_size);
        printf("[PanguLU Error] Please reduce 'init_options.nb' and try again. Exiting.\n");
        exit(1);
    }
    gessm_cuda<<<
        PANGULU_ICEIL(nb, pangulu_gpu_kernel_warp_per_block),
        pangulu_gpu_kernel_warp_per_block * PANGULU_WARP_SIZE,
        shared_memory_size>>>(
        nb,
        opdst->d_columnpointer, opdst->d_rowindex, (calculate_real_type *)opdst->d_value,
        opdiag->d_columnpointer, opdiag->d_rowindex, (calculate_real_type *)opdiag->d_value);

#ifdef PANGULU_PERF
    pangulu_platform_0201000_synchronize();
    global_stat.time_inner_kernel += pangulu_time_stop(&start);
#endif
    pangulu_cuda_download_block(nb, opdst);
}

void pangulu_platform_0201000_ssssm(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *op1,
    pangulu_storage_slot_t *op2,
    int tid)
{
#ifdef PANGULU_PERF
    struct timeval start;
    cudaDeviceSynchronize();
    pangulu_time_start(&start);
#endif

    if (!cublas_handle)
    {
        cublasCreate(&cublas_handle);
    }

    if(!d_ssssm_dense_buf_opdst){
        cudaMalloc(&d_ssssm_dense_buf_opdst, sizeof(calculate_type) * nb * nb);
        cudaMalloc(&d_ssssm_dense_buf_op1, sizeof(calculate_type) * nb * nb);
        cudaMalloc(&d_ssssm_dense_buf_op2, sizeof(calculate_type) * nb * nb);
    }

    if (opdst->brow_pos == opdst->bcol_pos)
    {
        pangulu_load_dense<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(nb, op1->d_columnpointer, op1->d_rowindex, (calculate_real_type *)op1->d_value, (calculate_real_type *)d_ssssm_dense_buf_op1);
        pangulu_load_dense<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(nb, op2->d_columnpointer, op2->d_rowindex, (calculate_real_type *)op2->d_value, (calculate_real_type *)d_ssssm_dense_buf_op2);
        pangulu_storage_slot_t *upper_diag;
        pangulu_storage_slot_t *lower_diag;
        if (opdst->is_upper)
        {
            upper_diag = opdst;
            lower_diag = opdst->related_block;
        }
        else
        {
            upper_diag = opdst->related_block;
            lower_diag = opdst;
        }
        pangulu_load_dense_getrf<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(
            nb,
            lower_diag->d_columnpointer, lower_diag->d_rowindex, (calculate_real_type *)lower_diag->d_value,
            upper_diag->d_rowpointer, upper_diag->d_columnindex, (calculate_real_type *)upper_diag->d_value,
            (calculate_real_type *)d_ssssm_dense_buf_opdst);

        calculate_type alpha = -1.0;
        calculate_type beta = 1.0;
#if defined(CALCULATE_TYPE_R64)
        cublasDgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            nb, nb, nb,
            &alpha,
            d_ssssm_dense_buf_op1, nb,
            d_ssssm_dense_buf_op2, nb,
            &beta,
            d_ssssm_dense_buf_opdst, nb);
#elif defined(CALCULATE_TYPE_R32)
        cublasSgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            nb, nb, nb,
            &alpha,
            d_ssssm_dense_buf_op1, nb,
            d_ssssm_dense_buf_op2, nb,
            &beta,
            d_ssssm_dense_buf_opdst, nb);
#elif defined(CALCULATE_TYPE_CR64)
        cublasZgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            nb, nb, nb,
            (cuDoubleComplex *)&alpha,
            (cuDoubleComplex *)d_ssssm_dense_buf_op1, nb,
            (cuDoubleComplex *)d_ssssm_dense_buf_op2, nb,
            (cuDoubleComplex *)&beta,
            (cuDoubleComplex *)d_ssssm_dense_buf_opdst, nb);
#elif defined(CALCULATE_TYPE_CR32)
        cublasCgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            nb, nb, nb,
            (cuFloatComplex *)&alpha,
            (cuFloatComplex *)d_ssssm_dense_buf_op1, nb,
            (cuFloatComplex *)d_ssssm_dense_buf_op2, nb,
            (cuFloatComplex *)&beta,
            (cuFloatComplex *)d_ssssm_dense_buf_opdst, nb);
#else
#error [PanguLU ERROR] Invalid CALCULATE_TYPE marco.
#endif

        pangulu_store_dense_getrf<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(
            nb,
            lower_diag->d_columnpointer, lower_diag->d_rowindex, (calculate_real_type *)lower_diag->d_value,
            upper_diag->d_rowpointer, upper_diag->d_columnindex, (calculate_real_type *)upper_diag->d_value,
            (calculate_real_type *)d_ssssm_dense_buf_opdst);
    }
    else
    {
#ifndef PANGULU_COMPLEX
        if ((op1->columnpointer[nb] == nb * nb) && (op2->columnpointer[nb] == nb * nb) && (opdst->columnpointer[nb] == nb * nb))
        {
            calculate_type alpha = -1.0;
            calculate_type beta = 1.0;
#if defined(CALCULATE_TYPE_R64)
            cublasDgemm(
                cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                nb, nb, nb,
                &alpha,
                op1->d_value, nb,
                op2->d_value, nb,
                &beta,
                opdst->d_value, nb);
#elif defined(CALCULATE_TYPE_R32)
            cublasSgemm(
                cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                nb, nb, nb,
                &alpha,
                op1->d_value, nb,
                op2->d_value, nb,
                &beta,
                opdst->d_value, nb);
#else
#error [PanguLU ERROR] Invalid CALCULATE_TYPE marco.
#endif
        }
        else
        {
#endif
            ssssm_cuda<<<
                nb,
                pangulu_gpu_kernel_warp_per_block * PANGULU_WARP_SIZE,
                sizeof(calculate_type) * nb>>>(
                nb,
                opdst->d_columnpointer, opdst->d_rowindex, (calculate_real_type *)opdst->d_value,
                op1->d_columnpointer, op1->d_rowindex, (calculate_real_type *)op1->d_value,
                op2->d_columnpointer, op2->d_rowindex, (calculate_real_type *)op2->d_value);
#ifndef PANGULU_COMPLEX
        }
#endif
    }

#ifdef PANGULU_PERF
    cudaDeviceSynchronize();
    global_stat.time_inner_kernel += pangulu_time_stop(&start);
#endif
}

// void pangulu_platform_0201000_hybrid_batched(
//     pangulu_inblock_idx nb,
//     pangulu_uint64_t ntask,
//     pangulu_task_t *tasks)
// {
//     for (pangulu_uint64_t itask = 0; itask < ntask; itask++)
//     {
//         switch (tasks[itask].kernel_id)
//         {
//         case PANGULU_TASK_GETRF:
//             pangulu_platform_0201000_getrf(nb, tasks[itask].opdst, 0);
//             break;
//         case PANGULU_TASK_TSTRF:
//             pangulu_platform_0201000_tstrf(nb, tasks[itask].opdst, tasks[itask].op1, 0);
//             break;
//         case PANGULU_TASK_GESSM:
//             pangulu_platform_0201000_gessm(nb, tasks[itask].opdst, tasks[itask].op1, 0);
//             break;
//         case PANGULU_TASK_SSSSM:
//             pangulu_platform_0201000_ssssm(nb, tasks[itask].opdst, tasks[itask].op1, tasks[itask].op2, 0);
//             break;
//         }
//     }
// }

__device__ pangulu_inblock_ptr
binarysearch_inblk_cuda(
    pangulu_inblock_idx *ridx,
    pangulu_int32_t left,
    pangulu_int32_t right,
    pangulu_inblock_idx target)
{
    pangulu_int32_t mid;
    while (left <= right)
    {
        mid = left + (right - left) / 2;
        if (ridx[mid] == target)
        {
            return mid;
        }
        else if (ridx[mid] > target)
        {
            right = mid - 1;
        }
        else
        {
            left = mid + 1;
        }
    }
    return 0xffffffff;
}

__global__ void store_csc_to_dense(
    pangulu_inblock_idx nb,
    pangulu_inblock_ptr *d_colptr,
    pangulu_inblock_idx *d_rowidx,
    calculate_type *d_value,
    calculate_type *dense)
{
    int col = blockIdx.x;
    if (col >= nb)
    {
        return;
    }
    for (int idx = col * nb + threadIdx.x; idx < (col + 1) * nb; idx += blockDim.x)
    {
        dense[idx] = 0.0;
    }
    // __syncthreads();
    for (int idx = d_colptr[col] + threadIdx.x; idx < d_colptr[col + 1]; idx += blockDim.x)
    {
        dense[col * nb + d_rowidx[idx]] = d_value[idx];
    }
}

__global__ void csc_add_dense(
    pangulu_inblock_idx nb,
    pangulu_inblock_ptr *d_colptr,
    pangulu_inblock_idx *d_rowidx,
    calculate_type *d_value,
    calculate_type *dense)
{
    int col = blockIdx.x;
    if (col >= nb)
    {
        return;
    }
    for (int idx = d_colptr[col] + threadIdx.x; idx < d_colptr[col + 1]; idx += blockDim.x)
    {
        d_value[idx] += dense[col * nb + d_rowidx[idx]];
    }
}

__global__ void csc_atomicadd_dense(
    pangulu_inblock_idx nb,
    pangulu_inblock_ptr *d_colptr,
    pangulu_inblock_idx *d_rowidx,
    calculate_type *d_value,
    calculate_type *dense)
{
    int col = blockIdx.x;
    if (col >= nb)
    {
        return;
    }
    for (int idx = d_colptr[col] + threadIdx.x; idx < d_colptr[col + 1]; idx += blockDim.x)
    {
        atomicAdd(&d_value[idx], dense[col * nb + d_rowidx[idx]]);
    }
}

__global__ void diag_add_dense(
    pangulu_inblock_idx nb,
    pangulu_inblock_ptr *d_colptr,
    pangulu_inblock_idx *d_rowidx,
    calculate_type *d_value_csc,
    pangulu_inblock_ptr *d_rowptr,
    pangulu_inblock_idx *d_colidx,
    calculate_type *d_value_csr,
    calculate_type *dense)
{
    int rc = blockIdx.x;
    if (rc >= nb)
    {
        return;
    }
    for (int idx = d_colptr[rc] + threadIdx.x; idx < d_colptr[rc + 1]; idx += blockDim.x)
    {
        d_value_csc[idx] += dense[rc * nb + d_rowidx[idx]];
    }
    for (int idx = d_rowptr[rc] + threadIdx.x; idx < d_rowptr[rc + 1]; idx += blockDim.x)
    {
        d_value_csr[idx] += dense[d_colidx[idx] * nb + rc];
    }
}

__global__ void diag_atomicadd_dense(
    pangulu_inblock_idx nb,
    pangulu_inblock_ptr *d_colptr,
    pangulu_inblock_idx *d_rowidx,
    calculate_type *d_value_csc,
    pangulu_inblock_ptr *d_rowptr,
    pangulu_inblock_idx *d_colidx,
    calculate_type *d_value_csr,
    calculate_type *dense)
{
    int rc = blockIdx.x;
    if (rc >= nb)
    {
        return;
    }
    for (int idx = d_colptr[rc] + threadIdx.x; idx < d_colptr[rc + 1]; idx += blockDim.x)
    {
        atomicAdd(&d_value_csc[idx], dense[rc * nb + d_rowidx[idx]]);
    }
    for (int idx = d_rowptr[rc] + threadIdx.x; idx < d_rowptr[rc + 1]; idx += blockDim.x)
    {
        atomicAdd(&d_value_csr[idx], dense[d_colidx[idx] * nb + rc]);
    }
}

__device__ pangulu_int32_t
get_task_id(
    pangulu_int32_t *blockmap,
    pangulu_int32_t left,
    pangulu_int32_t right,
    pangulu_int32_t target)
{
    pangulu_int32_t mid;
    while (left <= right)
    {
        mid = left + (right - left) / 2;
        if (blockmap[mid] > target)
        {
            right = mid - 1;
        }
        else
        {
            left = mid + 1;
        }
    }
    return left;
}

__global__ void trojan_horse_batched_kernel_cuda(
    pangulu_inblock_idx n,
    pangulu_uint64_t ntasks,
    pangulu_int32_t *d_task_types,
    pangulu_int32_t *d_task_block_ptr,
    pangulu_inblock_ptr **dd_rowptrc,
    pangulu_inblock_idx **dd_colidxc,
    calculate_type **dd_valuec,
    pangulu_inblock_ptr **dd_rowptrb,
    pangulu_inblock_idx **dd_colidxb,
    calculate_type **dd_valueb,
    pangulu_inblock_ptr **dd_rowptra,
    pangulu_inblock_idx **dd_colidxa,
    calculate_type **dd_valuea,
    calculate_type **dd_getrf_tag_double,
    pangulu_int32_t **dd_getrf_nnzu,
    pangulu_inblock_ptr **dd_getrf_csccolptrl_upperbound,
    pangulu_inblock_idx **dd_getrf_cscrowidxl_upperbound,
    pangulu_inblock_ptr **dd_getrf_csccolptru_upperbound,
    pangulu_inblock_idx **dd_getrf_cscrowidxu_upperbound,
    pangulu_inblock_ptr **dd_tstrf_a_valueidx,
    pangulu_inblock_ptr **dd_tstrf_l_valueidx)
{
    extern __shared__ char shared_memory[];
    pangulu_inblock_idx *s_idxa = (pangulu_inblock_idx *)shared_memory;
    calculate_type *s_dense = (calculate_type *)(shared_memory + sizeof(pangulu_inblock_idx) * n * (blockDim.x / PANGULU_WARP_SIZE));

    const int pangulu_gpu_kernel_warp_per_block = blockDim.x / PANGULU_WARP_SIZE;

    pangulu_int32_t task_id = get_task_id(d_task_block_ptr, 0, ntasks - 1, blockIdx.x);
    pangulu_int32_t task_type = d_task_types[task_id];
    pangulu_int32_t block_offset = 0;
    pangulu_int32_t nblock_for_task = 0;

    if (task_id == 0)
    {
        block_offset = blockIdx.x;
        nblock_for_task = d_task_block_ptr[0];
    }
    else
    {
        block_offset = blockIdx.x - d_task_block_ptr[task_id - 1];
        nblock_for_task = d_task_block_ptr[task_id] - d_task_block_ptr[task_id - 1];
    }

    pangulu_inblock_ptr *d_rowptrc = dd_rowptrc[task_id];
    pangulu_inblock_idx *d_colidxc = dd_colidxc[task_id];
    calculate_type *d_valuec = dd_valuec[task_id];
    pangulu_inblock_ptr *d_rowptrb = dd_rowptrb[task_id];
    pangulu_inblock_idx *d_colidxb = dd_colidxb[task_id];
    calculate_type *d_valueb = dd_valueb[task_id];
    pangulu_inblock_ptr *d_rowptra = dd_rowptra[task_id];
    pangulu_inblock_idx *d_colidxa = dd_colidxa[task_id];
    calculate_type *d_valuea = dd_valuea[task_id];

    if (task_type == PANGULU_TASK_SSSSM)
    {
        // pangulu_int32_t how_many_thread_a_col_need = PANGULU_MIN(PANGULU_THREAD_PER_BLOCK, PANGULU_ICEIL(dd_rowptrb[task_id][n], n) * TROJAN_HORSE_SSSSM_BATCHED_THREAD_PER_ELEM);
        // pangulu_int32_t how_many_col_each_block_can_process = PANGULU_THREAD_PER_BLOCK / how_many_thread_a_col_need;
        // pangulu_inblock_idx fst_row_this_block = block_offset * how_many_col_each_block_can_process;

        // if ((block_offset == 0) && (threadIdx.x == 0))
        // {
        //     d_rowptrc[0] = 0;
        //     d_rowptrb[0] = 0;
        //     d_rowptra[0] = 0;
        // }

        // const pangulu_inblock_idx row = fst_row_this_block + (threadIdx.x / how_many_thread_a_col_need);
        // const pangulu_inblock_idx thread_offset = threadIdx.x % how_many_thread_a_col_need;

        // if (row >= (fst_row_this_block + how_many_col_each_block_can_process))
        // {
        //     return;
        // }
        // if (row >= n)
        // {
        //     return;
        // }

        // if ((d_rowptrc[n] == (int)n * n) && (d_rowptrb[n] == (int)n * n) && (d_rowptra[n] == (int)n * n))
        // {
        //     for (pangulu_inblock_idx rowb = 0; rowb < n; rowb++)
        //     {
        //         calculate_type a_val = d_valuea[row * n + rowb];
        //         for (pangulu_inblock_idx colb = thread_offset; colb < n; colb += how_many_thread_a_col_need)
        //         {
        //             atomicAdd(&d_valuec[row * n + colb], -a_val * d_valueb[rowb * n + colb]);
        //         }
        //     }
        // }
        // else
        // {
        //     pangulu_inblock_ptr therowc = d_rowptrc[row];
        //     pangulu_inblock_ptr nextrowc = d_rowptrc[row + 1];

        //     pangulu_inblock_ptr therow = d_rowptra[row];
        //     pangulu_inblock_ptr nextrow = d_rowptra[row + 1];

        //     for (pangulu_inblock_ptr i = therow; i < nextrow; i++)
        //     {
        //         pangulu_inblock_idx cola = d_colidxa[i];
        //         calculate_type vala = d_valuea[i];

        //         pangulu_inblock_ptr therowb = d_rowptrb[cola];
        //         pangulu_inblock_ptr nextrowb = d_rowptrb[cola + 1];

        //         for (pangulu_inblock_ptr j = therowb + thread_offset; j < nextrowb; j += how_many_thread_a_col_need)
        //         {
        //             pangulu_inblock_idx colb = d_colidxb[j];
        //             pangulu_inblock_ptr flag = binarysearch_inblk_cuda(d_colidxc, therowc, nextrowc - 1, colb);
        //             if (flag != 0xffffffff)
        //             {
        //                 atomicAdd(&d_valuec[flag], -vala * d_valueb[j]);
        //             }
        //         }
        //     }
        // }

        pangulu_int32_t how_many_thread_a_col_need = PANGULU_MIN(PANGULU_THREAD_PER_BLOCK, PANGULU_ICEIL(dd_rowptrb[task_id][n], n) * TROJAN_HORSE_SSSSM_BATCHED_THREAD_PER_ELEM);
        pangulu_int32_t how_many_col_each_block_can_process = PANGULU_THREAD_PER_BLOCK / how_many_thread_a_col_need;
        pangulu_inblock_idx fst_row_this_block = block_offset * how_many_col_each_block_can_process;

        if ((block_offset == 0) && (threadIdx.x == 0))
        {
            d_rowptrc[0] = 0;
            d_rowptrb[0] = 0;
            d_rowptra[0] = 0;
        }

        const pangulu_inblock_idx row = fst_row_this_block + (threadIdx.x / how_many_thread_a_col_need);
        const pangulu_inblock_idx thread_offset = threadIdx.x % how_many_thread_a_col_need;

        if (row >= (fst_row_this_block + how_many_col_each_block_can_process))
        {
            return;
        }
        if (row >= n)
        {
            return;
        }

        if ((d_rowptrc[n] == (int)n * n) && (d_rowptrb[n] == (int)n * n) && (d_rowptra[n] == (int)n * n))
        {
            for (pangulu_inblock_idx rowb = 0; rowb < n; rowb++)
            {
                calculate_type a_val = d_valuea[row * n + rowb];
                for (pangulu_inblock_idx colb = thread_offset; colb < n; colb += how_many_thread_a_col_need)
                {
                    atomicAdd(&d_valuec[row * n + colb], -a_val * d_valueb[rowb * n + colb]);
                }
            }
        }
        else
        {
            if (how_many_col_each_block_can_process == 1)
            {
                pangulu_inblock_ptr therowc = d_rowptrc[row];
                pangulu_inblock_ptr nextrowc = d_rowptrc[row + 1];

                pangulu_inblock_ptr therow = d_rowptra[row];
                pangulu_inblock_ptr nextrow = d_rowptra[row + 1];

                for (pangulu_inblock_ptr idx = therowc + thread_offset; idx < nextrowc; idx += how_many_thread_a_col_need)
                {
                    pangulu_inblock_idx col = d_colidxc[idx];
                    s_dense[col] = 0;
                }

                __syncthreads();

                for (pangulu_inblock_ptr i = therow; i < nextrow; i++)
                {
                    pangulu_inblock_idx cola = d_colidxa[i];
                    calculate_type vala = d_valuea[i];

                    pangulu_inblock_ptr therowb = d_rowptrb[cola];
                    pangulu_inblock_ptr nextrowb = d_rowptrb[cola + 1];

                    for (pangulu_inblock_ptr j = therowb + thread_offset; j < nextrowb; j += how_many_thread_a_col_need)
                    {
                        pangulu_inblock_idx colb = d_colidxb[j];
                        // s_dense[colb] += vala * d_valueb[j];
                        atomicAdd(&s_dense[colb], vala * d_valueb[j]);
                        // pangulu_inblock_ptr flag = binarysearch_inblk_cuda(d_colidxc, therowc, nextrowc - 1, colb);
                        // if (flag != 0xffffffff)
                        // {
                        //     atomicAdd(&d_valuec[flag], -vala * d_valueb[j]);
                        // }
                    }
                }

                __syncthreads();

                for (pangulu_inblock_ptr idx = therowc + thread_offset; idx < nextrowc; idx += how_many_thread_a_col_need)
                {
                    pangulu_inblock_idx col = d_colidxc[idx];
                    atomicAdd(&d_valuec[idx], -s_dense[col]);
                }
            }
            else
            {
                pangulu_inblock_ptr therowc = d_rowptrc[row];
                pangulu_inblock_ptr nextrowc = d_rowptrc[row + 1];

                pangulu_inblock_ptr therow = d_rowptra[row];
                pangulu_inblock_ptr nextrow = d_rowptra[row + 1];

                for (pangulu_inblock_ptr i = therow; i < nextrow; i++)
                {
                    pangulu_inblock_idx cola = d_colidxa[i];
                    calculate_type vala = d_valuea[i];

                    pangulu_inblock_ptr therowb = d_rowptrb[cola];
                    pangulu_inblock_ptr nextrowb = d_rowptrb[cola + 1];

                    for (pangulu_inblock_ptr j = therowb + thread_offset; j < nextrowb; j += how_many_thread_a_col_need)
                    {
                        pangulu_inblock_idx colb = d_colidxb[j];
                        pangulu_inblock_ptr flag = binarysearch_inblk_cuda(d_colidxc, therowc, nextrowc - 1, colb);
                        if (flag != 0xffffffff)
                        {
                            atomicAdd(&d_valuec[flag], -vala * d_valueb[j]);
                        }
                    }
                }
            }
        }
    }
    // LU
    else if (task_type == PANGULU_TASK_GETRF)
    {
        const int tid = blockDim.x * block_offset + threadIdx.x;
        const int warpid = tid / PANGULU_WARP_SIZE;
        const int warp_tid = tid % PANGULU_WARP_SIZE;

        const int colidx = warpid;
        if (colidx >= n)
        {
            return;
        }

        const pangulu_inblock_ptr baseu_colidx = dd_getrf_csccolptru_upperbound[task_id][colidx];
        const pangulu_inblock_ptr baseu_colidx1 = dd_getrf_csccolptru_upperbound[task_id][colidx + 1];
        const pangulu_inblock_ptr basel_colidx = dd_getrf_csccolptrl_upperbound[task_id][colidx];
        const pangulu_inblock_ptr basel_colidx1 = dd_getrf_csccolptrl_upperbound[task_id][colidx + 1];

        // step one
        for (pangulu_inblock_ptr j = baseu_colidx; j < baseu_colidx1 - 1; j++)
        {
            const pangulu_inblock_idx rowidx = dd_getrf_cscrowidxu_upperbound[task_id][j];
            // busy-wait until nnzu[rowidx] == 0
            do
            {
                __threadfence();
            } while (dd_getrf_nnzu[task_id][rowidx] != 0);

            calculate_type bcast_value = dd_getrf_tag_double[task_id][colidx * n + rowidx];
            for (pangulu_inblock_ptr i = dd_getrf_csccolptrl_upperbound[task_id][rowidx] + 1 + warp_tid; i < dd_getrf_csccolptrl_upperbound[task_id][rowidx + 1]; i += PANGULU_WARP_SIZE)
            {
                const int lrowindex = dd_getrf_cscrowidxl_upperbound[task_id][i];
                const int lcolindex = rowidx;
                dd_getrf_tag_double[task_id][colidx * n + lrowindex] -= dd_getrf_tag_double[task_id][lcolindex * n + lrowindex] * bcast_value;
                // atomicAdd(&d_dense_tag_double[colidx * n + lrowindex], -d_dense_tag_double[lcolindex * n + lrowindex] * bcast_value);
            }
        }

        // __threadfence();
        //  step two
        calculate_type diag_value_inv = 1.0 / dd_getrf_tag_double[task_id][colidx * n + colidx];
        for (pangulu_inblock_ptr i = basel_colidx + warp_tid + 1; i < dd_getrf_csccolptrl_upperbound[task_id][colidx + 1]; i += PANGULU_WARP_SIZE)
        {
            const int lrowindex = dd_getrf_cscrowidxl_upperbound[task_id][i];
            dd_getrf_tag_double[task_id][colidx * n + lrowindex] = dd_getrf_tag_double[task_id][colidx * n + lrowindex] * diag_value_inv;
        }

        if (!warp_tid)
        {
            dd_getrf_nnzu[task_id][colidx] = 0;
        }
    }

    else if (task_type == PANGULU_TASK_GESSM)
    {
        pangulu_inblock_idx colidx = block_offset;
        if (colidx >= n)
        {
            return;
        }

        pangulu_inblock_ptr *a_columnpointer = d_rowptrc;
        pangulu_inblock_idx *a_rowindex = d_colidxc;
        calculate_type *a_value = d_valuec;
        pangulu_inblock_ptr *l_columnpointer = d_rowptrb;
        pangulu_inblock_idx *l_rowindex = d_colidxb;
        calculate_type *l_value = d_valueb;

        pangulu_inblock_ptr cola1 = a_columnpointer[colidx];
        pangulu_inblock_ptr cola2 = a_columnpointer[colidx + 1];
        if (cola2 == cola1)
        {
            return;
        }

        for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
        {
            s_idxa[i] = a_rowindex[cola1 + i];
            s_dense[s_idxa[i]] = a_value[cola1 + i];
        }
        __syncthreads();

        for (pangulu_int64_t i = cola1, t = 0; i < cola2; i++, t++)
        {
            pangulu_int64_t rowa = s_idxa[t];
            calculate_type vala = s_dense[s_idxa[t]];
            pangulu_int64_t coll1 = l_columnpointer[rowa];
            pangulu_int64_t coll2 = l_columnpointer[rowa + 1];
            for (pangulu_int64_t j = coll1 + threadIdx.x, p = threadIdx.x; j < coll2; j += blockDim.x, p += blockDim.x)
            {
                s_dense[l_rowindex[j]] -= vala * l_value[j];
            }
            __syncthreads();
        }

        for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
        {
            a_value[cola1 + i] = s_dense[s_idxa[i]];
        }
        //__syncthreads();
    }
    else if (task_type == PANGULU_TASK_TSTRF)
    {
        pangulu_inblock_ptr *a_columnpointer = d_rowptrc;
        pangulu_inblock_idx *a_rowindex = d_colidxc;
        calculate_type *a_value = d_valuec;
        pangulu_inblock_ptr *a_valueidx = dd_tstrf_a_valueidx[task_id];

        pangulu_inblock_ptr *l_columnpointer = d_rowptrb;
        pangulu_inblock_idx *l_rowindex = d_colidxb;
        calculate_type *l_value = d_valueb;
        // pangulu_inblock_ptr* l_valueidx = dd_tstrf_l_valueidx[task_id];

        pangulu_inblock_idx colidx = block_offset;
        if (colidx >= n)
        {
            return;
        }

        pangulu_inblock_ptr cola1 = a_columnpointer[colidx];
        pangulu_inblock_ptr cola2 = a_columnpointer[colidx + 1];
        if (cola2 == cola1)
        {
            return;
        }

        for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
        {
            s_idxa[i] = a_rowindex[cola1 + i];
            s_dense[s_idxa[i]] = a_value[a_valueidx[cola1 + i]];
        }
        __syncthreads();

        for (pangulu_int64_t i = cola1, t = 0; i < cola2; i++, t++)
        {
            pangulu_int64_t rowa = s_idxa[t];
            pangulu_int64_t coll1 = binarysearch_inblk_cuda(l_rowindex, l_columnpointer[rowa], l_columnpointer[rowa + 1] - 1, rowa);
            pangulu_int64_t coll2 = l_columnpointer[rowa + 1];

            calculate_type vala;
            if ((threadIdx.x / 32) == 0)
            {
                vala = s_dense[s_idxa[t]];
                // vala /= l_value[l_valueidx[coll1]];
                vala /= l_value[coll1];
                s_dense[s_idxa[t]] = vala;
                __syncthreads();
            }
            else
            {
                __syncthreads();
                __threadfence_block();
                vala = s_dense[s_idxa[t]];
            }

            for (pangulu_int64_t j = coll1 + 1 + threadIdx.x, p = threadIdx.x; j < coll2; j += blockDim.x, p += blockDim.x)
            {
                // update a's value;
                // s_dense[l_rowindex[j]] -= vala * l_value[l_valueidx[j]];
                s_dense[l_rowindex[j]] -= vala * l_value[j];
            }
            __syncthreads();
        }

        for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
        {
            a_value[a_valueidx[cola1 + i]] = s_dense[s_idxa[i]];
        }
    }
}

void pangulu_platform_0201000_hybrid_batched(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t *tasks)
{
#define PANGULU_REMALLOC_HOST(ptr, type)     \
    ptr = (type)(info_pool_h + pool_offset); \
    pool_offset += sizeof(*ptr) * ntask;
#define PANGULU_REMALLOC_DEVICE(ptr, type)   \
    ptr = (type)(info_pool_d + pool_offset); \
    pool_offset += sizeof(*ptr) * ntask;
#define PANGULU_HYBRID_PARAM_SIZE \
    (3 * (sizeof(pangulu_inblock_ptr *) + sizeof(pangulu_inblock_idx *) + sizeof(calculate_type *)) + sizeof(pangulu_int32_t) + sizeof(pangulu_int32_t) + sizeof(calculate_type *) + sizeof(pangulu_int32_t *) + sizeof(pangulu_inblock_ptr *) * 4 + sizeof(pangulu_inblock_idx *) * 2)

    if(ntask == 1){
        switch (tasks[0].kernel_id)
        {
        case PANGULU_TASK_GETRF:
            pangulu_platform_0201000_getrf(nb, tasks[0].opdst, 0);
            break;
        case PANGULU_TASK_TSTRF:
            pangulu_platform_0201000_tstrf(nb, tasks[0].opdst, tasks[0].op1, 0);
            break;
        case PANGULU_TASK_GESSM:
            pangulu_platform_0201000_gessm(nb, tasks[0].opdst, tasks[0].op1, 0);
            break;
        case PANGULU_TASK_SSSSM:
            pangulu_platform_0201000_ssssm(nb, tasks[0].opdst, tasks[0].op1, tasks[0].op2, 0);
            break;
        }
        return;
    }
    
    double dense_threshold = 1;
    pangulu_int64_t last_sparse_task_idx = ntask - 1;
    for (pangulu_int64_t i = 0; i <= last_sparse_task_idx; i++)
    {
        if (tasks[i].kernel_id == PANGULU_TASK_SSSSM)
        {
            if ((tasks[i].opdst->bcol_pos == tasks[i].opdst->brow_pos) ||
                (tasks[i].op1->columnpointer[nb] + tasks[i].op2->columnpointer[nb] > dense_threshold * nb * nb * 2))
            {
                pangulu_task_t tmp;
                memcpy(&tmp, &tasks[i], sizeof(pangulu_task_t));
                memcpy(&tasks[i], &tasks[last_sparse_task_idx], sizeof(pangulu_task_t));
                memcpy(&tasks[last_sparse_task_idx], &tmp, sizeof(pangulu_task_t));
                last_sparse_task_idx--;
                i--;
            }
        }
    }
    pangulu_int64_t dense_ssssm_cnt = ntask - last_sparse_task_idx - 1;
    ntask = last_sparse_task_idx + 1;

    if (ntask != 0)
    {
        if (task_pointer_buf_capacity < ntask)
        {
            if (info_pool_h)
            {
                pangulu_free(__FILE__, __LINE__, info_pool_h);
            }
            info_pool_h = (char *)pangulu_malloc(__FILE__, __LINE__, ntask * PANGULU_HYBRID_PARAM_SIZE);
            if (info_pool_d)
            {
                pangulu_platform_0201000_free(info_pool_d);
            }
            pangulu_platform_0201000_malloc((void **)&(info_pool_d), ntask * PANGULU_HYBRID_PARAM_SIZE);

            task_pointer_buf_capacity = ntask;
        }

        unsigned long long pool_offset = 0;
        PANGULU_REMALLOC_HOST(h_task_types, pangulu_int32_t *);
        PANGULU_REMALLOC_HOST(h_task_block_ptr, pangulu_int32_t *);
        PANGULU_REMALLOC_HOST(hd_rowptrc, pangulu_inblock_ptr **);
        PANGULU_REMALLOC_HOST(hd_colidxc, pangulu_inblock_idx **);
        PANGULU_REMALLOC_HOST(hd_valuec, calculate_type **);
        PANGULU_REMALLOC_HOST(hd_rowptrb, pangulu_inblock_ptr **);
        PANGULU_REMALLOC_HOST(hd_colidxb, pangulu_inblock_idx **);
        PANGULU_REMALLOC_HOST(hd_valueb, calculate_type **);
        PANGULU_REMALLOC_HOST(hd_rowptra, pangulu_inblock_ptr **);
        PANGULU_REMALLOC_HOST(hd_colidxa, pangulu_inblock_idx **);
        PANGULU_REMALLOC_HOST(hd_valuea, calculate_type **);
        PANGULU_REMALLOC_HOST(hd_getrf_tag_double, calculate_type **);
        PANGULU_REMALLOC_HOST(hd_getrf_nnzu, pangulu_int32_t **);
        PANGULU_REMALLOC_HOST(hd_getrf_csccolptrl_upperbound, pangulu_inblock_ptr **);
        PANGULU_REMALLOC_HOST(hd_getrf_cscrowidxl_upperbound, pangulu_inblock_idx **);
        PANGULU_REMALLOC_HOST(hd_getrf_csccolptru_upperbound, pangulu_inblock_ptr **);
        PANGULU_REMALLOC_HOST(hd_getrf_cscrowidxu_upperbound, pangulu_inblock_idx **);
        PANGULU_REMALLOC_HOST(hd_tstrf_a_valueidx, pangulu_inblock_ptr **);
        PANGULU_REMALLOC_HOST(hd_tstrf_l_valueidx, pangulu_inblock_ptr **);

        pool_offset = 0;
        PANGULU_REMALLOC_DEVICE(d_task_types, pangulu_int32_t *);
        PANGULU_REMALLOC_DEVICE(d_task_block_ptr, pangulu_int32_t *);
        PANGULU_REMALLOC_DEVICE(dd_rowptrc, pangulu_inblock_ptr **);
        PANGULU_REMALLOC_DEVICE(dd_colidxc, pangulu_inblock_idx **);
        PANGULU_REMALLOC_DEVICE(dd_valuec, calculate_type **);
        PANGULU_REMALLOC_DEVICE(dd_rowptrb, pangulu_inblock_ptr **);
        PANGULU_REMALLOC_DEVICE(dd_colidxb, pangulu_inblock_idx **);
        PANGULU_REMALLOC_DEVICE(dd_valueb, calculate_type **);
        PANGULU_REMALLOC_DEVICE(dd_rowptra, pangulu_inblock_ptr **);
        PANGULU_REMALLOC_DEVICE(dd_colidxa, pangulu_inblock_idx **);
        PANGULU_REMALLOC_DEVICE(dd_valuea, calculate_type **);
        PANGULU_REMALLOC_DEVICE(dd_getrf_tag_double, calculate_type **);
        PANGULU_REMALLOC_DEVICE(dd_getrf_nnzu, pangulu_int32_t **);
        PANGULU_REMALLOC_DEVICE(dd_getrf_csccolptrl_upperbound, pangulu_inblock_ptr **);
        PANGULU_REMALLOC_DEVICE(dd_getrf_cscrowidxl_upperbound, pangulu_inblock_idx **);
        PANGULU_REMALLOC_DEVICE(dd_getrf_csccolptru_upperbound, pangulu_inblock_ptr **);
        PANGULU_REMALLOC_DEVICE(dd_getrf_cscrowidxu_upperbound, pangulu_inblock_idx **);
        PANGULU_REMALLOC_DEVICE(dd_tstrf_a_valueidx, pangulu_inblock_ptr **);
        PANGULU_REMALLOC_DEVICE(dd_tstrf_l_valueidx, pangulu_inblock_ptr **);

        pangulu_uint64_t getrf_idx = 0;
        for (pangulu_uint64_t i = 0; i < ntask; i++)
        {
            h_task_types[i] = tasks[i].kernel_id;
            if (tasks[i].kernel_id == PANGULU_TASK_GETRF)
            {
                getrf_idx++;
            }
        }

        if (getrf_idx > getrf_buffer_cap)
        {
            getrf_indeces_cap = getrf_idx;
            getrf_indeces = (pangulu_uint64_t *)pangulu_realloc(__FILE__, __LINE__, getrf_indeces, sizeof(pangulu_uint64_t) * getrf_indeces_cap);

            getrf_buffer_cap = getrf_indeces_cap;
            if (d_getrf_tag_buffer)
            {
                cudaFree(d_getrf_tag_buffer);
            }
            cudaMalloc(&d_getrf_tag_buffer, sizeof(calculate_type) * nb * nb * getrf_buffer_cap);
            if (!d_getrf_tag_buffer)
            {
                printf("cudaMalloc error NULL (2), allocationg %lld B\n", sizeof(calculate_type) * nb * nb * getrf_buffer_cap);
            }
        }

        getrf_idx = 0;
        for (pangulu_uint64_t i = 0; i < ntask; i++)
        {
            if (tasks[i].kernel_id == PANGULU_TASK_TSTRF)
            {
                hd_rowptrc[i] = tasks[i].opdst->d_rowpointer;
                hd_colidxc[i] = tasks[i].opdst->d_columnindex;
                hd_valuec[i] = tasks[i].opdst->d_value;
                hd_tstrf_a_valueidx[i] = tasks[i].opdst->d_idx_of_csc_value_for_csr;

                if (tasks[i].op1->is_upper == 0)
                {
                    tasks[i].op1 = tasks[i].op1->related_block;
                }
                hd_rowptrb[i] = tasks[i].op1->d_rowpointer;
                hd_colidxb[i] = tasks[i].op1->d_columnindex;
                hd_valueb[i] = tasks[i].op1->d_value;
            }
            else
            {
                hd_rowptrc[i] = tasks[i].opdst->d_columnpointer;
                hd_colidxc[i] = tasks[i].opdst->d_rowindex;
                hd_valuec[i] = tasks[i].opdst->d_value;

                if (tasks[i].kernel_id == PANGULU_TASK_GESSM)
                {
                    if (tasks[i].op1->is_upper == 1)
                    {
                        tasks[i].op1 = tasks[i].op1->related_block;
                    }
                    hd_rowptrb[i] = tasks[i].op1->d_columnpointer;
                    hd_colidxb[i] = tasks[i].op1->d_rowindex;
                    hd_valueb[i] = tasks[i].op1->d_value;
                }

                if (tasks[i].kernel_id == PANGULU_TASK_SSSSM)
                {
                    hd_rowptrb[i] = tasks[i].op1->d_columnpointer;
                    hd_colidxb[i] = tasks[i].op1->d_rowindex;
                    hd_valueb[i] = tasks[i].op1->d_value;
                    hd_rowptra[i] = tasks[i].op2->d_columnpointer;
                    hd_colidxa[i] = tasks[i].op2->d_rowindex;
                    hd_valuea[i] = tasks[i].op2->d_value;
                }

                if (tasks[i].kernel_id == PANGULU_TASK_GETRF)
                {
                    // hd_getrf_nnzu[i] = tasks[i].opdst->d_nnzu;
                    // hd_getrf_csccolptrl_upperbound[i] = tasks[i].opdst->d_csccolptrl_upperbound;
                    // hd_getrf_cscrowidxl_upperbound[i] = tasks[i].opdst->d_cscrowidxl_upperbound;
                    // hd_getrf_csccolptru_upperbound[i] = tasks[i].opdst->d_csccolptru_upperbound;
                    // hd_getrf_cscrowidxu_upperbound[i] = tasks[i].opdst->d_cscrowidxu_upperbound;

                    // hd_getrf_tag_double[i] = d_getrf_tag_buffer + getrf_idx * nb * nb;
                    // pangulu_load_dense<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(
                    //     nb,
                    //     tasks[i].opdst->d_columnpointer,
                    //     tasks[i].opdst->d_rowindex,
                    //     tasks[i].opdst->d_value,
                    //     hd_getrf_tag_double[i]
                    // );
                    // getrf_indeces[getrf_idx] = i;
                    // getrf_idx++;
                }
            }

            if (tasks[i].kernel_id == PANGULU_TASK_GETRF)
            {
                h_task_block_ptr[i] = PANGULU_ICEIL(nb, pangulu_gpu_kernel_warp_per_block);
            }
            else if (tasks[i].kernel_id == PANGULU_TASK_TSTRF)
            {
                h_task_block_ptr[i] = nb;
            }
            else if (tasks[i].kernel_id == PANGULU_TASK_GESSM)
            {
                h_task_block_ptr[i] = nb;
            }
            else if (tasks[i].kernel_id == PANGULU_TASK_SSSSM)
            {
                pangulu_int32_t how_many_thread_a_col_need = PANGULU_MIN(PANGULU_THREAD_PER_BLOCK, PANGULU_ICEIL(tasks[i].op1->columnpointer[nb], nb) * TROJAN_HORSE_SSSSM_BATCHED_THREAD_PER_ELEM);
                pangulu_int32_t how_many_col_each_block_can_process = PANGULU_THREAD_PER_BLOCK / how_many_thread_a_col_need;
                pangulu_int32_t need_block = PANGULU_ICEIL(nb, how_many_col_each_block_can_process);
                h_task_block_ptr[i] = need_block;
            }
        }

        for (int i = 1; i < ntask; i++)
        {
            h_task_block_ptr[i] += h_task_block_ptr[i - 1];
        }

        pangulu_platform_0201000_memcpy(
            info_pool_d, info_pool_h,
            ntask * PANGULU_HYBRID_PARAM_SIZE,
            0);

        size_t shared_memory_size = (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * nb * pangulu_gpu_kernel_warp_per_block;
        if (shared_memory_size > pangulu_gpu_shared_mem_size)
        {
            printf("[PanguLU Error] Requested shared memory size %d bytes exceeds the maximum limit of %d bytes.\n", shared_memory_size, pangulu_gpu_shared_mem_size);
            printf("[PanguLU Error] Please reduce 'init_options.nb' and try again. Exiting.\n");
            exit(1);
        }
        trojan_horse_batched_kernel_cuda<<<
            h_task_block_ptr[ntask - 1],
            PANGULU_THREAD_PER_BLOCK,
            shared_memory_size>>>(
            nb,
            ntask,
            d_task_types,
            d_task_block_ptr,
            dd_rowptrc,
            dd_colidxc,
            dd_valuec,
            dd_rowptrb,
            dd_colidxb,
            dd_valueb,
            dd_rowptra,
            dd_colidxa,
            dd_valuea,
            dd_getrf_tag_double,
            dd_getrf_nnzu,
            dd_getrf_csccolptrl_upperbound,
            dd_getrf_cscrowidxl_upperbound,
            dd_getrf_csccolptru_upperbound,
            dd_getrf_cscrowidxu_upperbound,
            dd_tstrf_a_valueidx,
            dd_tstrf_l_valueidx);

        pangulu_platform_0201000_synchronize();

        for (int i_getrf = 0; i_getrf < getrf_idx; i_getrf++)
        {
            int itask = getrf_indeces[i_getrf];
            pangulu_store_dense<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(
                nb,
                tasks[itask].opdst->d_columnpointer,
                tasks[itask].opdst->d_rowindex,
                tasks[itask].opdst->d_value,
                d_getrf_tag_buffer + i_getrf * nb * nb);
        }
        for (int i = 0; i < ntask; i++)
        {
            if (tasks[i].kernel_id != PANGULU_TASK_SSSSM)
            {
                pangulu_cuda_download_block(nb, tasks[i].opdst);
            }
        }
        pangulu_platform_0201000_synchronize();
        cudaError_t err = cudaGetLastError();
        if (err)
        {
            printf("error2 : %s\n", cudaGetErrorString(err));
        }
    }

    if (dense_ssssm_cnt)
    {
        if (dense_ssssm_cnt > dense_task_indeces_cap)
        {
            dense_task_indeces_cap = dense_ssssm_cnt;

            if (d_dense_buffer)
            {
                cudaFree(d_dense_buffer);
            }
            cudaMalloc(&d_dense_buffer, sizeof(calculate_type) * nb * nb * dense_task_indeces_cap * 3);
            if (!d_dense_buffer)
            {
                printf("cudaMalloc error NULL (1), allocationg %lld B\n", sizeof(calculate_type) * nb * nb * dense_task_indeces_cap * 3);
            }

            hd_ssssm_batch_ptr = (calculate_type **)pangulu_realloc(__FILE__, __LINE__, hd_ssssm_batch_ptr, sizeof(calculate_type *) * dense_task_indeces_cap * 3);

            if (dd_ssssm_batch_ptr)
            {
                cudaFree(dd_ssssm_batch_ptr);
            }
            cudaMalloc(&dd_ssssm_batch_ptr, sizeof(calculate_type *) * dense_task_indeces_cap * 3);
            if (!dd_ssssm_batch_ptr)
            {
                printf("cudaMalloc error NULL (dd_ssssm_batch_ptr), allocationg %lld B\n", sizeof(calculate_type *) * dense_task_indeces_cap * 3);
            }
        }

        for (pangulu_int64_t i = ntask; i < ntask + dense_ssssm_cnt; i++)
        {
            pangulu_int64_t dense_task_idx = i - ntask;
            store_csc_to_dense<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(nb, tasks[i].op1->d_columnpointer, tasks[i].op1->d_rowindex, tasks[i].op1->d_value, d_dense_buffer + ((3 * dense_task_idx) * nb * nb));
            store_csc_to_dense<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(nb, tasks[i].op2->d_columnpointer, tasks[i].op2->d_rowindex, tasks[i].op2->d_value, d_dense_buffer + ((3 * dense_task_idx + 1) * nb * nb));
            hd_ssssm_batch_ptr[0 * dense_ssssm_cnt + dense_task_idx] = d_dense_buffer + ((3 * dense_task_idx + 0) * nb * nb);
            hd_ssssm_batch_ptr[1 * dense_ssssm_cnt + dense_task_idx] = d_dense_buffer + ((3 * dense_task_idx + 1) * nb * nb);
            hd_ssssm_batch_ptr[2 * dense_ssssm_cnt + dense_task_idx] = d_dense_buffer + ((3 * dense_task_idx + 2) * nb * nb);
        }

        if (!cublas_handle)
        {
            cublasCreate(&cublas_handle);
        }

        cudaMemcpy(dd_ssssm_batch_ptr, hd_ssssm_batch_ptr, sizeof(calculate_type *) * dense_ssssm_cnt * 3, cudaMemcpyHostToDevice);

#ifdef PANGULU_PERF
        global_stat.kernel_cnt++;
        struct timeval start;
        pangulu_platform_0201000_synchronize();
        pangulu_time_start(&start);
#endif

        calculate_type alpha = -1.0;
        calculate_type beta = 0.0;
#if defined(CALCULATE_TYPE_R64)
        cublasDgemmBatched(
            cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            nb, nb, nb,
            &alpha,
            dd_ssssm_batch_ptr + 0 * dense_ssssm_cnt, nb,
            dd_ssssm_batch_ptr + 1 * dense_ssssm_cnt, nb,
            &beta,
            dd_ssssm_batch_ptr + 2 * dense_ssssm_cnt, nb,
            dense_ssssm_cnt);
#elif defined(CALCULATE_TYPE_R32)
        cublasSgemmBatched(
            cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            nb, nb, nb,
            &alpha,
            dd_ssssm_batch_ptr + 0 * dense_ssssm_cnt, nb,
            dd_ssssm_batch_ptr + 1 * dense_ssssm_cnt, nb,
            &beta,
            dd_ssssm_batch_ptr + 2 * dense_ssssm_cnt, nb,
            dense_ssssm_cnt);
#endif

        // for(pangulu_int64_t i = ntask; i < ntask + dense_ssssm_cnt; i++){
        //     cublasDgemm(
        //         cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        //         nb, nb, nb,
        //         &alpha,
        //         hd_ssssm_batch_ptr[0 * dense_ssssm_cnt + i - ntask], nb,
        //         hd_ssssm_batch_ptr[1 * dense_ssssm_cnt + i - ntask], nb,
        //         &beta,
        //         hd_ssssm_batch_ptr[2 * dense_ssssm_cnt + i - ntask], nb);
        // }

#ifdef PANGULU_PERF
        pangulu_platform_0201000_synchronize();
        global_stat.time_inner_kernel += pangulu_time_stop(&start);
#endif

        for (pangulu_int64_t i = ntask; i < ntask + dense_ssssm_cnt; i++)
        {
            if (tasks[i].opdst->brow_pos == tasks[i].opdst->bcol_pos)
            {
                pangulu_storage_slot_t *lower_diag;
                pangulu_storage_slot_t *upper_diag;
                if (tasks[i].opdst->is_upper)
                {
                    upper_diag = tasks[i].opdst;
                    lower_diag = tasks[i].opdst->related_block;
                }
                else
                {
                    upper_diag = tasks[i].opdst->related_block;
                    lower_diag = tasks[i].opdst;
                }
                diag_atomicadd_dense<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(
                    nb,
                    lower_diag->d_columnpointer,
                    lower_diag->d_rowindex,
                    lower_diag->d_value,
                    upper_diag->d_rowpointer,
                    upper_diag->d_columnindex,
                    upper_diag->d_value,
                    hd_ssssm_batch_ptr[2 * dense_ssssm_cnt + (i - ntask)]);
            }
            else
            {
                csc_atomicadd_dense<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(
                    nb,
                    tasks[i].opdst->d_columnpointer,
                    tasks[i].opdst->d_rowindex,
                    tasks[i].opdst->d_value,
                    hd_ssssm_batch_ptr[2 * dense_ssssm_cnt + (i - ntask)]);
            }
        }
    }

#undef PANGULU_REMALLOC_HOST
#undef PANGULU_REMALLOC_DEVICE
}

void pangulu_platform_0201000_ssssm_batched(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t *tasks)
{
    // for (pangulu_uint64_t itask = 0; itask < ntask; itask++)
    // {
    //     pangulu_platform_0201000_ssssm(nb, tasks[itask].opdst, tasks[itask].op1, tasks[itask].op2, 0);
    // }
    pangulu_platform_0201000_hybrid_batched(nb, ntask, tasks);
}

#else

void pangulu_platform_0201000_getrf(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    int tid)
{
}
void pangulu_platform_0201000_tstrf(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *opdiag,
    int tid)
{
}
void pangulu_platform_0201000_gessm(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *opdiag,
    int tid)
{
}
void pangulu_platform_0201000_ssssm(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *op1,
    pangulu_storage_slot_t *op2,
    int tid)
{
}

void pangulu_platform_0201000_ssssm_batched(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t *tasks)
{
}

void pangulu_platform_0201000_hybrid_batched(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t *tasks)
{
}

#endif

void pangulu_platform_0201000_spmv(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *a,
    calculate_type *x,
    calculate_type *y)
{
}

void pangulu_platform_0201000_vecadd(
    pangulu_int64_t length,
    calculate_type *bval,
    calculate_type *xval)
{
}

void pangulu_platform_0201000_sptrsv(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *s,
    calculate_type *xval,
    pangulu_int64_t uplo)
{
}
