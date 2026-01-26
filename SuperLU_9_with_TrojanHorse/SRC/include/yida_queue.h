//本文件是yida_queue.h，用于构建生产者+消费者线程的环状队列。作者李易达
#ifndef YIDA_QUEUE
#define YIDA_QUEUE
#include <semaphore.h>
#include <stdlib.h>
#include <string.h>

typedef struct Queue{
    sem_t full,empty,mutex;
    unsigned head,tail,length,elemsize;
    void* data;
}Queue;

Queue* queue_init(unsigned length, unsigned elemSize){
    Queue* ret = (Queue*)malloc(sizeof(Queue));
    if(ret==NULL)return NULL;
    sem_init(&ret->full,1,0);
    sem_init(&ret->empty,1,length);
    sem_init(&ret->mutex,1,1);
    ret->head = 0;
    ret->tail = 0;
    ret->length = length;
    ret->elemsize = elemSize;
    ret->data = malloc(elemSize*length);
    if(ret->data==NULL)return NULL;
    return ret;
}

//返回元素地址，需快用
void* queue_pop_fast(Queue* queue){
    unsigned retIdx;
    sem_wait(&queue->full);
    sem_wait(&queue->mutex);
    retIdx = queue->head;
    queue->head = (queue->head+1)%queue->length;
    sem_post(&queue->mutex);
    sem_post(&queue->empty);
    return queue->data+queue->elemsize*retIdx;
}

//返回拷贝的元素地址，可长期使用，需free
void* queue_pop(Queue* queue){
    void* ret = malloc(queue->elemsize);
    sem_wait(&queue->full);
    sem_wait(&queue->mutex);
    memcpy(ret,queue->data+queue->elemsize*queue->head,queue->elemsize);
    queue->head = (queue->head+1)%queue->length;
    sem_post(&queue->mutex);
    sem_post(&queue->empty);
    return ret;
}

//elem指向要插入的元素
void queue_push(Queue* queue,void* elem){
    sem_wait(&queue->empty);
    sem_wait(&queue->mutex);
    memcpy(queue->data+queue->elemsize*queue->tail,elem,queue->elemsize);
    queue->tail = (queue->tail+1)%queue->length;
    sem_post(&queue->mutex);
    sem_post(&queue->full);
}
#endif
