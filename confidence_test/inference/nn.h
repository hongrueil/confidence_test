#ifndef NN_H_
#define NN_H_


#include <string.h>
#include <stdio.h>
#include "lib.h"

#define CPU

/************ define mask************/
#define bit_mask_16     0XFFFF
#define bit_mask_12     0XFFF0
#define bit_mask_8      0XFF00
#define bit_mask_4      0XF000
#define bit_mask_1      0X8000
#define bit_mask_2      0XC000
#define bit_mask_3      0XE000


typedef struct __tensor{
    uint16_t ndim;
    uint16_t dims[4];
    uint16_t bw;
    const fixed *data;
} Tensor;

typedef struct __buffer{
    uint16_t ndim;
    uint16_t dims[4];
    uint16_t bw;
    fixed *data;
} Buffer;







static void swap_buffer(Buffer *a, Buffer*b){
    uint16_t temp;
    temp = a->ndim;
    a->ndim = b->ndim;
    b->ndim = temp;

    uint16_t arr[4];
    memcpy(arr, a->dims, 4 * sizeof(uint16_t));
    memcpy(a->dims, b->dims, 4 * sizeof(uint16_t));
    memcpy(b->dims, arr, 4 * sizeof(uint16_t));

    _q15 *dptr;
    dptr = a->data;
    a->data = b->data;
    b->data = dptr;
}

Buffer *fc(const Buffer *input, const Tensor *weight, const Tensor *bias, Buffer *output);


/**
 * weight dims:
 *      [0]: output channels 
 *      [1]: input channels
 *      [2]: filter width
 *      [3]: filter height
*/
//fixed *conv2d_3x3(const fixed *input, const fixed *filter, fixed *output);
Buffer *conv2d_fir(const Buffer *input, const Tensor *weight, const Tensor *bias, Buffer *output);
Buffer *conv2d_fir_quant(const Buffer *input, const Tensor *weight, const Tensor *bias, Buffer *output, int *quant_list);
Buffer *conv2d_fir_dis(const Buffer *input, const Tensor *weight, const Tensor *bias, Buffer *output, int *quant_list);
Buffer *conv2d_fir_quant_norm(const Buffer *input, const Tensor *weight, const Tensor *bias, Buffer *output, double *norm_l1_list, int in_num);
Buffer *conv2d_fir_quant_norm_l2(const Buffer *input, const Tensor *weight, const Tensor *bias, Buffer *output, double *norm_l1_list, int in_num);
//Buffer *conv2d_im2col(const Buffer *input, const Tensor *weight, const Tensor *bias, Buffer *output);
Buffer *maxpool(const Buffer *input, Buffer *output, uint16_t pool_size);

void weight_norm(const Tensor *weight, double *norm_l1_list);

static Buffer *relu(Buffer *input, Buffer *output){
    /* metadata */
    output->ndim = input->ndim;
    memcpy(output->dims, input->dims, sizeof(uint16_t) * input->ndim);

    /* relu */
    unsigned size = 1;
    for(unsigned i = 0; i < input->ndim; ++i){
        size *= input->dims[i];
    }

    for(unsigned i = 0; i < size; ++i){
//        if(i == 0) msp_benchmarkStart(MSP_BENCHMARK_BASE, 64);
        output->data[i] = input->data[i] < 0 ? 0 : input->data[i];
//        if(i == 0) printf("timer relu: %lu * %d\n", msp_benchmarkStop(MSP_BENCHMARK_BASE), size);
    }

    return output;
}

static Buffer *flatten(Buffer *tensor){
    unsigned acc = 1;
    for(unsigned i = 0; i < tensor->ndim; ++i){
        acc *= tensor->dims[i];
    }
    // 因為是直接把 weight 存成一維 array，所以 flatten 其實不用改，只要改 dimension 資訊就好
    tensor->dims[0] = acc;
    tensor->ndim = 1;

    return tensor;
}

/* simulation unpack */
static void unpack(const Tensor *src){
    uint32_t size = 1;
    for(unsigned i = 0; i < src->ndim; ++i){
        size *= src->dims[i];
    }
    size /= 4;

    _q15 a[4] = {};
    for(unsigned i = 0; i < size; ++i){
//        if(i == 0) msp_benchmarkStart(MSP_BENCHMARK_BASE, 64);
        a[0] = src->data[i] & 0xF000;
        a[1] = (src->data[i]  & 0x0F00) << 4;
        a[2] = (src->data[i]  & 0x00F0) << 8;
        a[3] = (src->data[i]  & 0x000F) << 12;
//        if(i == 0) printf("timer sim unpack: %lu * %lu\n", msp_benchmarkStop(MSP_BENCHMARK_BASE), size);
    }
    a[0] = a[0] + a[1] + a[2] + a[3];
}

/*********** quant data to 8 bits ************/
static void quant(Buffer *buf, int q){
    if (q == 16) {
        q = 0XFFFF; 
        //return;
    } 
    else if (q == 12) q = 0XFFF0;
    else if (q == 8) q = 0XFF00;
    else if (q == 4) q = 0XF000;
    else if (q == 2) q = 0XC000;

    unsigned size = 1;
    // 算出整個 buffer 的資料量 --> size
    for(unsigned i = 0; i < buf->ndim; ++i){
        size *= buf->dims[i];
    }

//    msp_benchmarkStart(MSP_BENCHMARK_BASE, 64);
    for(unsigned i = 0; i < size; ++i){

        //printf("before = %d ",buf->data[i]);
        buf->data[i] &= (fixed)(q);
        //printf("after = %d\n",buf->data[i]);
    }
//    printf("timer quant: %lu\n", msp_benchmarkStop(MSP_BENCHMARK_BASE));
}

static void dump_buf(Buffer *buf){
    unsigned size = 1;
    for(unsigned i = 0; i < buf->ndim; ++i){
        size *= buf->dims[i];
    }
    for(unsigned i = 0; i < size; ++i){
       printf("%d ",buf->data[i]);
    }
    printf("\n------------------------------\n");
}


#endif