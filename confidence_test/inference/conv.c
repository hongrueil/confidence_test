#include "nn.h"



Buffer *conv2d_fir(const Buffer *input, const Tensor *weight, const Tensor *bias, Buffer *output) {
    output->ndim = input->ndim;
    output->dims[0] = weight->dims[0];
    output->dims[1] = input->dims[1] - (weight->dims[2] - 1);
    output->dims[2] = input->dims[2] - (weight->dims[3] - 1);

    const unsigned input_channel = input->dims[0];
    const unsigned input_map_size = input->dims[1] * input->dims[2];
    const unsigned filter_size = weight->dims[2] * weight->dims[3];
    const unsigned output_channel = output->dims[0];
    const unsigned output_map_size = output->dims[1] * output->dims[2];

#ifdef CPU
    for(unsigned out_ch = 0; out_ch < output_channel; ++out_ch){ 
        for(unsigned out_i = 0; out_i < output_map_size; ++out_i){

            _iq31 acc = 0;
            const unsigned out_row = (out_i ) / output->dims[2];
            const unsigned out_col = (out_i ) % output->dims[1];


            for(unsigned in_ch = 0; in_ch < input_channel; ++in_ch){
//                if(out_i + out_ch + in_ch == 0) msp_benchmarkStart(MSP_BENCHMARK_BASE, 64);
                /* todo: speedup*/
                for(unsigned row = 0; row < weight->dims[2]; ++row){
                    for(unsigned col = 0; col < weight->dims[3]; ++col){
//                          acc += (input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)]) * (weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col]);

                        acc = __saturated_add_iq31(acc, __q15mpyl(input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)], ((weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col]& 0XFFFFFF00))));
                        //acc = __saturated_add_iq31(acc, __q15mpyl(input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)], ((weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col]))));

                        
                        //printf("%d  %d   %d\n",(int32_t)(weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col]),weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col],(int32_t)( weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col] & 0XFFFFF000));
                    }
                }
//                if(out_i + out_ch + in_ch == 0) printf("timer CONV2D: %lu * %d * %d * %d * %d\n", msp_benchmarkStop(MSP_BENCHMARK_BASE), input_channel, output_channel, output->dims[1], output->dims[2]);
            }

            output->data[out_ch * output_map_size + out_i] = acc >> 16; // 這裡會有問題嗎 ? 好像不會
            //printf("acc = %d, acc shift = %d, output = %d, type output = %d\n",acc, acc >> 16, output->data[out_ch * output_map_size + out_i],(int32_t)output->data[out_ch * output_map_size + out_i]);
           // output->data[out_ch * output_map_size + out_i] = acc;
            output->data[out_ch * output_map_size + out_i] =__saturated_add_q15(output->data[out_ch * output_map_size + out_i], (bias->data[out_ch] & 0XFF00));
            //output->data[out_ch * output_map_size + out_i] =__saturated_add_q15(output->data[out_ch * output_map_size + out_i], bias->data[out_ch]);
            //output->data[out_ch * output_map_size + out_i] >>= 2;

        }
    }
#endif

}


Buffer *conv2d_fir_quant(const Buffer *input, const Tensor *weight, const Tensor *bias, Buffer *output, int *quant_list) {
    output->ndim = input->ndim;
    output->dims[0] = weight->dims[0];
    output->dims[1] = input->dims[1] - (weight->dims[2] - 1);
    output->dims[2] = input->dims[2] - (weight->dims[3] - 1);

    const unsigned input_channel = input->dims[0];
    const unsigned input_map_size = input->dims[1] * input->dims[2];
    const unsigned filter_size = weight->dims[2] * weight->dims[3];
    const unsigned output_channel = output->dims[0];
    const unsigned output_map_size = output->dims[1] * output->dims[2];
    int quant_index = 0;

#ifdef CPU
    for(unsigned out_ch = 0; out_ch < output_channel; ++out_ch){  //最外這圈在換 kernel
        if (out_ch == quant_list[quant_index]) { //out_ch == quant_list[quant_index]
            //printf("quant ch = %d, ",out_ch);
            //quant
            quant_index++;
            for(unsigned out_i = 0; out_i < output_map_size; ++out_i){

                _iq31 acc = 0;

                const unsigned out_row = (out_i ) / output->dims[2];
                const unsigned out_col = (out_i ) % output->dims[1];


                for(unsigned in_ch = 0; in_ch < input_channel; ++in_ch){
    //                if(out_i + out_ch + in_ch == 0) msp_benchmarkStart(MSP_BENCHMARK_BASE, 64);
                    for(unsigned row = 0; row < weight->dims[2]; ++row){
                        for(unsigned col = 0; col < weight->dims[3]; ++col){   
                            // quant
                            acc = __saturated_add_iq31(acc, __q15mpyl((input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)] & 0XFFFFFF00), ((weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col] & 0XFFFFFF00)))); // 因為這個 macro 會先 extend 成 36bit, 所以後面的 & 也要改成 32bit
                        }
                    }
                }
                output->data[out_ch * output_map_size + out_i] = acc >> 16; 
                // quant
                output->data[out_ch * output_map_size + out_i] =__saturated_add_q15(output->data[out_ch * output_map_size + out_i], bias->data[out_ch] & 0XFF00); 
                //output->data[out_ch * output_map_size + out_i] >>= 2;

            }
          
        } else {
            //no quant
            for(unsigned out_i = 0; out_i < output_map_size; ++out_i){
                _iq31 acc = 0;
                const unsigned out_row = (out_i ) / output->dims[2];
                const unsigned out_col = (out_i ) % output->dims[1];


                for(unsigned in_ch = 0; in_ch < input_channel; ++in_ch){
    //                if(out_i + out_ch + in_ch == 0) msp_benchmarkStart(MSP_BENCHMARK_BASE, 64);
                    for(unsigned row = 0; row < weight->dims[2]; ++row){
                        for(unsigned col = 0; col < weight->dims[3]; ++col){
                            // no quant
                            acc = __saturated_add_iq31(acc, __q15mpyl(input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)], ((weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col] & 0XFFFFFFFF))));
                        }
                    }                
                }

                output->data[out_ch * output_map_size + out_i] = acc >> 16; 
                output->data[out_ch * output_map_size + out_i] =__saturated_add_q15(output->data[out_ch * output_map_size + out_i], bias->data[out_ch] & 0XFFFF);
                //output->data[out_ch * output_map_size + out_i] >>= 2;
            }
        }
       

//
    }
#endif

}


Buffer *conv2d_fir_quant_norm(const Buffer *input, const Tensor *weight, const Tensor *bias, Buffer *output, double *norm_l1_list, int in_num) {
    output->ndim = input->ndim;
    output->dims[0] = weight->dims[0];
    output->dims[1] = input->dims[1] - (weight->dims[2] - 1);
    output->dims[2] = input->dims[2] - (weight->dims[3] - 1);

    const unsigned input_channel = input->dims[0];
    const unsigned input_map_size = input->dims[1] * input->dims[2];
    const unsigned filter_size = weight->dims[2] * weight->dims[3];
    const unsigned output_channel = output->dims[0];
    const unsigned output_map_size = output->dims[1] * output->dims[2];
    

#ifdef CPU
    for(unsigned out_ch = 0; out_ch < output_channel; ++out_ch){  //最外這圈在換 kernel
    long long int Norm_L1 = 0;
        for(unsigned out_i = 0; out_i < output_map_size; ++out_i){

            _iq31 acc1 = 0;
            _iq31 acc2 = 0;
            fixed quant_output = 0;
           // int64_t  acc3 = 0;
            const unsigned out_row = (out_i ) / output->dims[2];
            const unsigned out_col = (out_i ) % output->dims[1];


            for(unsigned in_ch = 0; in_ch < input_channel; ++in_ch){
//                if(out_i + out_ch + in_ch == 0) msp_benchmarkStart(MSP_BENCHMARK_BASE, 64);
                /* todo: speedup*/
                for(unsigned row = 0; row < weight->dims[2]; ++row){
                    for(unsigned col = 0; col < weight->dims[3]; ++col){
                            // unquant
                            acc1 = __saturated_add_iq31(acc1, __q15mpyl(input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)], ((weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col] & 0XFFFFFFFF))));
                            // quant
                            acc2 = __saturated_add_iq31(acc2, __q15mpyl((input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)] & 0XFFFFFF00), ((weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col] & 0XFFFFFF00)))); // 因為這個 macro 會先 extend 成 36bit, 所以後面的 & 也要改成 32bit
                    }
                }
            
            }


            output->data[out_ch * output_map_size + out_i] = acc1 >> 16; 
            quant_output = acc2 >>16;

            // no quant
            output->data[out_ch * output_map_size + out_i] =__saturated_add_q15(output->data[out_ch * output_map_size + out_i], bias->data[out_ch] & 0XFFFF);

            // quant
            quant_output =__saturated_add_q15(quant_output, bias->data[out_ch] & 0XFF00);

            
            //output->data[out_ch * output_map_size + out_i] >>= 2;
            //quant_output >>= 2;

            
            //quant norm
            //long long int diff = output->data[out_ch * output_map_size + out_i] - quant_output;
            
            //drop norm
            long long int diff = output->data[out_ch * output_map_size + out_i];

            
            Norm_L1 += abs(diff);
            //if (out_ch <= 0) printf("diff = %lld, Norm_now = %lld\n",diff, Norm_L1);
            
      

        }
//        if (out_ch <= 40) {
            //printf("Norm of filter %d = %lld\n",out_ch, Norm_L1);
            norm_l1_list[out_ch] += (double)Norm_L1 / in_num;
            //printf("Norm l1_list[%d] = %lf\n",out_ch, norm_l1_list[out_ch]);

//        }

    }
#endif

}

Buffer *conv2d_fir_quant_norm_l2(const Buffer *input, const Tensor *weight, const Tensor *bias, Buffer *output, double *norm_l1_list, int in_num) {
    output->ndim = input->ndim;
    output->dims[0] = weight->dims[0];
    output->dims[1] = input->dims[1] - (weight->dims[2] - 1);
    output->dims[2] = input->dims[2] - (weight->dims[3] - 1);

    const unsigned input_channel = input->dims[0];
    const unsigned input_map_size = input->dims[1] * input->dims[2];
    const unsigned filter_size = weight->dims[2] * weight->dims[3];
    const unsigned output_channel = output->dims[0];
    const unsigned output_map_size = output->dims[1] * output->dims[2];
    

#ifdef CPU
    for(unsigned out_ch = 0; out_ch < output_channel; ++out_ch){  //最外這圈在換 kernel
    long long int Norm_L2 = 0;
        for(unsigned out_i = 0; out_i < output_map_size; ++out_i){

            _iq31 acc1 = 0;
            _iq31 acc2 = 0;
            fixed quant_output = 0;
           // int64_t  acc3 = 0;
            const unsigned out_row = (out_i ) / output->dims[2];
            const unsigned out_col = (out_i ) % output->dims[1];


            for(unsigned in_ch = 0; in_ch < input_channel; ++in_ch){
//                if(out_i + out_ch + in_ch == 0) msp_benchmarkStart(MSP_BENCHMARK_BASE, 64);
                /* todo: speedup*/
                for(unsigned row = 0; row < weight->dims[2]; ++row){
                    for(unsigned col = 0; col < weight->dims[3]; ++col){
                            // unquant
                            acc1 = __saturated_add_iq31(acc1, __q15mpyl(input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)], ((weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col] & 0XFFFFFFFF))));
                            // quant
                            acc2 = __saturated_add_iq31(acc2, __q15mpyl((input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)] & 0XFFFFFF00), ((weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col] & 0XFFFFFF00)))); // 因為這個 macro 會先 extend 成 36bit, 所以後面的 & 也要改成 32bit
                    }
                }
            
            }


            output->data[out_ch * output_map_size + out_i] = acc1 >> 16; 
            quant_output = acc2 >>16;

            // no quant
            output->data[out_ch * output_map_size + out_i] =__saturated_add_q15(output->data[out_ch * output_map_size + out_i], bias->data[out_ch] & 0XFFFF);

            // quant
            quant_output =__saturated_add_q15(quant_output, bias->data[out_ch] & 0XFF00);

            
            //output->data[out_ch * output_map_size + out_i] >>= 2;
            //quant_output >>= 2;

            
            //quant norm
            long long int diff = output->data[out_ch * output_map_size + out_i] - quant_output;
            long long  int diff_2 = diff * diff;
            

            
            
            
            //drop norm
            // long long int diff = output->data[out_ch * output_map_size + out_i];
            // long long  int diff_2 = diff * diff;
            
            Norm_L2 += diff_2;
            //if (out_ch == 0) printf("Norm L2 = %lld\n",Norm_L2);
            
      

        }

        double sqrt_val = 0;
        sqrt_val = sqrt((double)Norm_L2);

        //if (out_ch == 0)printf("sqrt val = %lf\n",sqrt_val);
        norm_l1_list[out_ch] += (double)sqrt_val / in_num;
        //printf("Norm l1_list[%d] = %lf\n",out_ch, norm_l1_list[out_ch]);



    }
#endif

}


Buffer *conv2d_fir_dis(const Buffer *input, const Tensor *weight, const Tensor *bias, Buffer *output, int *quant_list) {
    output->ndim = input->ndim;
    output->dims[0] = weight->dims[0];
    output->dims[1] = input->dims[1] - (weight->dims[2] - 1);
    output->dims[2] = input->dims[2] - (weight->dims[3] - 1);

    const unsigned input_channel = input->dims[0];
    const unsigned input_map_size = input->dims[1] * input->dims[2];
    const unsigned filter_size = weight->dims[2] * weight->dims[3];
    const unsigned output_channel = output->dims[0];
    const unsigned output_map_size = output->dims[1] * output->dims[2];
    int quant_index = 0;

#ifdef CPU
    for(unsigned out_ch = 0; out_ch < output_channel; ++out_ch){  //最外這圈在換 kernel
        if (out_ch == quant_list[quant_index]) {
            //printf("now out_ch = %d\n",out_ch);
            //quant
            quant_index++;
            for(unsigned out_i = 0; out_i < output_map_size; ++out_i){

                _iq31 acc = 0;

                const unsigned out_row = (out_i ) / output->dims[2];
                const unsigned out_col = (out_i ) % output->dims[1];


                for(unsigned in_ch = 0; in_ch < input_channel; ++in_ch){
    //                if(out_i + out_ch + in_ch == 0) msp_benchmarkStart(MSP_BENCHMARK_BASE, 64);
                    for(unsigned row = 0; row < weight->dims[2]; ++row){
                        for(unsigned col = 0; col < weight->dims[3]; ++col){   
                            // discard
                            acc = 0;
                        }
                    }
                }
                output->data[out_ch * output_map_size + out_i] = acc >> 16; 
                // quant
                //output->data[out_ch * output_map_size + out_i] =__saturated_add_q15(output->data[out_ch * output_map_size + out_i], bias->data[out_ch] & 0XF000); 
                //output->data[out_ch * output_map_size + out_i] >>= 2;

            }
          
        } else {
            //no quant
            for(unsigned out_i = 0; out_i < output_map_size; ++out_i){
                _iq31 acc = 0;
                const unsigned out_row = (out_i ) / output->dims[2];
                const unsigned out_col = (out_i ) % output->dims[1];


                for(unsigned in_ch = 0; in_ch < input_channel; ++in_ch){
    //                if(out_i + out_ch + in_ch == 0) msp_benchmarkStart(MSP_BENCHMARK_BASE, 64);
                    for(unsigned row = 0; row < weight->dims[2]; ++row){
                        for(unsigned col = 0; col < weight->dims[3]; ++col){
                            // no quant
                            acc = __saturated_add_iq31(acc, __q15mpyl(input->data[in_ch * input_map_size + (out_row + row) * input->dims[2] + (out_col + col)], ((weight->data[(out_ch * input_channel + in_ch) * filter_size + row * weight->dims[2] + col] & 0XFFFFFF00))));
                        }
                    }                
                }

                output->data[out_ch * output_map_size + out_i] = acc >> 16; 
                output->data[out_ch * output_map_size + out_i] =__saturated_add_q15(output->data[out_ch * output_map_size + out_i], bias->data[out_ch] & 0XFF00);
                //output->data[out_ch * output_map_size + out_i] >>= 2;
            }
        }
       

//
    }
#endif

}

void weight_norm(const Tensor *weight, double *norm_l1_list) {
    const unsigned input_channel = weight->dims[1];
    const unsigned output_channel = weight->dims[0];
    const unsigned filter_size = weight->dims[2] * weight->dims[3];


    for (unsigned out_ch = 0; out_ch < output_channel; ++out_ch) { //for each kernel
        long long int Norm_L1 = 0;
        for (unsigned in_ch = 0; in_ch < input_channel; ++in_ch) {
            for (unsigned f_i = 0; f_i < filter_size; ++f_i) {
                Norm_L1 += abs(weight -> data[(out_ch * input_channel + in_ch)* filter_size + f_i] );
                if (out_ch == 0) printf("weight = %d\n", weight -> data[(out_ch * input_channel + in_ch)* filter_size + f_i]);
                
            }
        }
        norm_l1_list[out_ch] = Norm_L1;
    }


}