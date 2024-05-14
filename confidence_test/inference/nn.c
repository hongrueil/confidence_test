#include "nn.h"


Buffer *maxpool(const Buffer *input, Buffer *output, const uint16_t pool_size){
    int stride = pool_size;

    /* reshape buffer */
    output->ndim = input->ndim;
    output->dims[0] = input->dims[0]; // input -> dims[0] 看起來是 input channel 
    output->dims[1] = input->dims[1] / pool_size;
    output->dims[2] = input->dims[2] / pool_size;

    /* max pooling */
//    const unsigned input_channel_size = input->dims[0];
//    const unsigned input_map_size = input->dims[0];
    const unsigned output_channel = output->dims[0];
    const unsigned output_map_size = output->dims[1] * output->dims[2];


    for(unsigned out_ch = 0; out_ch < output_channel; ++out_ch){
        for(unsigned out_i = 0; out_i < output_map_size; ++out_i){
//            if(out_i + out_ch == 0) msp_benchmarkStart(MSP_BENCHMARK_BASE, 64);
            _q15 maxi = INT16_MIN;
            unsigned out_row = (out_i) / output->dims[2];
            unsigned out_col = (out_i) % output->dims[1];


            for(unsigned row = 0; row < pool_size; ++row){
                for(unsigned col = 0; col < pool_size; ++col){
                    _q15 cmp = input->data[out_ch * input->dims[2] * input->dims[2] + (out_row * stride + row) * input->dims[2] + (out_col * stride + col)];
                    maxi = maxi < cmp ? cmp : maxi;
                }
            }

            output->data[out_ch * output_map_size + out_i] = maxi;
//            if(out_i + out_ch == 0) printf("timer maxpool: %lu * %d * %d\n", msp_benchmarkStop(MSP_BENCHMARK_BASE), output_channel, output_map_size);
        }
    }

    return output;
}