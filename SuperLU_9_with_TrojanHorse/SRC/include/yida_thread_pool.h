//本文件是yida_thread_pool.h，创建了线程池及其管理函数。作者李易达
#ifndef YIDA_THREAD_POOL
#define YIDA_THREAD_POOL
#include <stdio.h>
#define __USE_GNU
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include "yida_intstack.h"
struct Thread_Pool;

typedef struct _thread{
    volatile int state; //0是停止状态；1是就绪或运行状态
    int (*funct)(void*); //线程当前要执行的函数(start_routine)
    void* param; //参数(arg)
    pthread_t pthread; //POSIX线程
    int tid; //在线程池中的线程号
    struct Thread_Pool* in_which_pool; //所在线程池
    int bind_core_id;
}thread;

typedef struct Thread_Pool{
    Stack* idle; //闲置线程栈（存储池内tid号）
    thread* threads; //所有线程（线程号从1开始依次顺延）
}Thread_Pool;

// 本头文件内部函数，用于回收运行完的线程
int _recycle_thread(int tid, Thread_Pool* pool){
    pool->threads[tid-1].state = 0; //停
    return stack_push(pool->idle,tid); //收
}

void bind_to_core(int core)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);

    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0)
    {
        // printf("err1 = %d\n", err);
        perror("pthread_setaffinity_np error");
    }
}

int cuda_device = 0;

// 本头文件内部函数，空闲状态，是线程池每个线程的入口函数
void* _idle(void* _thread){
    thread* t = (thread*)_thread;
    Thread_Pool* pool = t->in_which_pool;
    int tid = t->tid;
    bind_to_core(t->bind_core_id);
    int cuda_dev_cnt;
    cudaGetDeviceCount(&cuda_dev_cnt);
    cudaSetDevice(cuda_device%cuda_dev_cnt);
    while(1){
        while((t->state)==0); //实时监测状态是否就绪
        int (*funct)(thread*) = (int (*)(thread*))t->funct; //线程池被使用时，需要调用的函数
        funct(t); //启动它
        _recycle_thread(tid,pool); //回收线程
    }
}

// 对外接口，用于创建线程池。参数为线程池容量，返回线程池，失败时返回NULL.
Thread_Pool* create_pool(int n_threads, int first_bind_core){
    Thread_Pool* ret = (Thread_Pool*)malloc(sizeof(Thread_Pool));
    if(ret==NULL)return NULL;
    ret->idle = stack_init(n_threads);
    ret->threads = (thread*)malloc(n_threads*sizeof(thread));
    for(int i=0;i<n_threads;i++){
        ret->idle->data[i]=n_threads-i;
        ret->threads[i].tid=i+1;
        ret->threads[i].in_which_pool = ret;
        ret->threads[i].bind_core_id = first_bind_core+i;
    }
    for(int i=0;i<n_threads;i++){
        ret->threads[i].state = 0;
        if(pthread_create(&(ret->threads[i].pthread),NULL,_idle,&(ret->threads[i]))!=0){
            return NULL;
        }
    }
    ret->idle->top=n_threads;
    return ret;
}

// 对外接口，在线程池中申请线程，返回线程在这个线程池中的tid号，从1开始。
// startNow参数置0就暂不启动，置1就立即启动。对于可预知线程入口的情况，预设置入口可以进一步加速。
int alloc_thread(Thread_Pool* pool,int (*function)(thread*),void* params, int startNow){
    int tid;
    while(stack_pop(pool->idle,&(tid))!=0);
    thread* t = &pool->threads[tid-1];
    t->funct = (int (*)(void*))function;
    t->param = params;
    t->state = startNow;
    return tid;
}

// 对外接口，启动一个线程。用于申请线程时没启动的情况。
int start_thread(int tid, Thread_Pool* pool){
    pool->threads[tid-1].state = 1;
    return 0;
}

// 对外接口，等待线程结束。
int join_thread(int tid, Thread_Pool* pool){
    while(pool->threads[tid-1].state);
    return 0;
}

#endif
