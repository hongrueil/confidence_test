#include "nn.h"

Buffer *fc(const Buffer *input, const Tensor *weight, const Tensor *bias, Buffer *output) {
    /* reshape buffer */
    output->ndim = input->ndim;
    output->dims[0] = weight->dims[0];

    /* vector-matrix multiplication */
    const unsigned output_size = output->dims[0];
    const unsigned input_size  = input->dims[0];

    /* CPU */
#ifdef CPU
    for(unsigned out_i = 0; out_i < output_size; ++out_i){
//        if(out_i == 0) msp_benchmarkStart(MSP_BENCHMARK_BASE, 64);
        /* todo: speedup */
        _iq31 acc = 0;
        for(unsigned in_i = 0; in_i < input_size; ++in_i){
//            acc += input->data[in_i] * weight->data[input_size * out_i + in_i];
            acc = __saturated_add_iq31(acc, __q15mpyl(input->data[in_i], (weight->data[input_size * out_i + in_i] & 0XFFFFFF00)));
            //acc = __saturated_add_iq31(acc, __q15mpyl(input->data[in_i], (weight->data[input_size * out_i + in_i])));
        }
        
        output->data[out_i] = acc >> 16; //turn output into 16bit
//        output->data[out_i] = acc;
        output->data[out_i] = __saturated_add_q15(output->data[out_i], bias->data[out_i] & 0XFF00);
        //output->data[out_i] = __saturated_add_q15(output->data[out_i], bias->data[out_i] );
        //output->data[out_i] >>= 2;
        // test
        //output->data[out_i] = output->data[out_i] < 0 ? 0 : output->data[out_i];
//        if(out_i == 0) printf("timer fc: %lu * %d\n", msp_benchmarkStop(MSP_BENCHMARK_BASE), weight->dims[0]);
    }
#endif
    return output;

}