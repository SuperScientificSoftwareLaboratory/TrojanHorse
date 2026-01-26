//本文件是yida_intstack.h，创建了一个整数栈类型。作者李易达
#ifndef YIDA_INTSTACK
#define YIDA_INTSTACK
#include <semaphore.h>
#include <stdlib.h>
typedef struct {
    int* data;
    int top;
    int length;
    sem_t mutex;
}Stack;

Stack* stack_init(int length){
    Stack* ret = (Stack*)malloc(sizeof(Stack));
    if(ret==NULL)return NULL;
    ret->data = (int*)malloc(length*sizeof(int));
    ret->length = length;
    sem_init(&ret->mutex,1,1);
    return ret;
}

int stack_push(Stack* stk, int num){
    sem_wait(&stk->mutex);
    if(stk->top>=stk->length){
        sem_post(&stk->mutex);
        return 1;
    }
    stk->data[stk->top]=num;
    stk->top++;
    sem_post(&stk->mutex);
    return 0;
}

int stack_pop(Stack* stk, int* num){
    sem_wait(&stk->mutex);
    if(stk->top<=0){
        sem_post(&stk->mutex);
        return 1;
    }
    stk->top --;
    *num = stk->data[stk->top];
    sem_post(&stk->mutex);
    return 0;
}
#endif
