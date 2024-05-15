#include <stdlib.h>
#include <stdbool.h>
#include "inference/nn.h"
/************* input **************/
/*
input data is from: https://github.com/pjreddie/mnist-csv-png/blob/master/process_mnist.py  
which has 10000 test data from mnist

*/



/********** include model *************/
//#include "dataset/mnist/input.h"
#include "model/hawaii/model.h"

#define mean 0.1307
#define sd 0.3081


/******* define what you want to do *******/
#define drop
//#define conv
//#define norm
//#define norm_l2

typedef struct res {
    bool res;
    int core_margin;
    int pred;
    int ans;
    int pred_list[10];
}res;


Buffer data_input = {
    .ndim = 3,
    .dims = {1, 28, 28},
    .bw = 0,
    .data = NULL
    //.data = (fixed *)data_input_raw[0]
};


/********** buffer A **********/
fixed bufferA_data[11600] = {0};
Buffer bufferA_tensor = {
    .ndim = 0,
    .dims = {0},
    .bw = 0,
    .data = (fixed *)bufferA_data
};

/********** buffer B **********/
fixed bufferB_data[11600] = {0};
Buffer bufferB_tensor = {
    .ndim = 0,
    .dims = {0},
    .bw = 0,
    .data = (fixed *)bufferB_data
};


int main(int argc, char *argv[]) {
    int right = 0;

    Buffer *bufferA, *bufferB;
    bufferA = &bufferA_tensor;
    bufferB = &bufferB_tensor;



    // for select quantization channel
    int quant_2d_list[6][40] = { {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, }, {23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, }, {23, 27, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, }, {11, 15, 23, 27, 29, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, }, {3, 4, 8, 9, 11, 15, 23, 27, 28, 29, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, }, {0, 3, 4, 5, 7, 8, 9, 11, 14, 15, 16, 19, 20, 21, 23, 24, 27, 28, 29, 33, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, } };
    
    // for counting the norm of each filter in conv2
    double norm_l1[40] = {0};

    res result_list[6][10000];
    int accuracy[6];
    int next_level_list[10000];
    int next_level_idx = 0;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 10000; ++j) {
            result_list[i][j].res = false;
            result_list[i][j].core_margin = 0;
        }
    }


    int bit_length;
    int in_num;
    int step_len;
    int step_start;
    int select_num;

// define bit length
    bit_length = atoi(argv[1]);
// define sample num
    in_num = atoi(argv[2]);
// use for set combination step
    step_len = atoi(argv[3]);
// use for set step start
    step_start = atoi(argv[4]);
// num for qunat list selection
    select_num = atoi(argv[5]);
    
    int *quant_list;
    //int *quant_list = &quant_2d_list[select_num][0];


    // /******** show quant list*********/
    // for (int i = 0; i < 40; ++i) printf("%d, ",quant_list[i]);




    FILE *fp = fopen("./dataset/input/mnist_test.csv", "r");
    if (fp == NULL) {
        fprintf(stderr, "fopen() failed.\n");
        exit(EXIT_FAILURE);
    }

    char str[2400];
    char *token;
    float input[28*28];
    fixed data_input_raw[28 * 28];


    int class_list[10] = {0};
    int wrong_list[10] = {0};

    for (int tt = 0; tt < 6; tt++) {

        
        // set fp back to the head of file
        rewind(fp); 
        // select quant list
        // if (tt == 0) quant_list = &quant_2d_list[tt][0];
        // else {
        //     quant_list = &quant_2d_list[4][0]; //drop 50%
        // }
        quant_list = &quant_2d_list[tt][0];
        

        // initialize
        right = 0;    
        next_level_idx = 0;

        printf("%d'th quant, quant list = ", tt);
        for (int i = 0; i < 40; ++i) {
            
            printf("%d, ",quant_list[i]); 
            if (quant_list[i] == -1) break; 
        }
        printf("\n");


        // iter for all sample input
        for (int i = 0; i < in_num; ++i) {

            int ans;

            /****************** load input from csv **********************/

            int index = 0;
            fgets(str, 2400, fp);
            token = strtok(str, ","); 
            ans = atoi(token); //the answer of this input
            
            // load csv which spilt by ','
            while (token != NULL) {
                //printf("%s ", token);
                token = strtok(NULL, ",");
                input[index] = (((atof(token) / 255) - mean) / sd);
                //input[index] = ((atof(token) / 255));
                data_input_raw[index] = _Q15_i(input[index]);
                index++;
            }
            /**************************************************************/


        //initial
            data_input.data = (fixed *)data_input_raw;




        // load input into buffer A
            bufferA->ndim = data_input.ndim;
            memcpy(bufferA->dims, data_input.dims, 6);
            memcpy(bufferA->data, data_input.data, 28*28*sizeof(_q15));

        // inference
            //int bit_length  = 8;
            quant(bufferA, bit_length);
            conv2d_fir(bufferA, &conv1_w, &conv1_b, bufferB); swap_buffer(bufferA, bufferB);
            

            quant(bufferA, bit_length);
            maxpool(bufferA, bufferB, 2); swap_buffer(bufferA, bufferB);
        
            quant(bufferA, bit_length);

        #ifdef quant
            conv2d_fir_quant(bufferA, &conv2_w, &conv2_b, bufferB, quant_list); swap_buffer(bufferA, bufferB);
        
        #endif

        #ifdef norm
            conv2d_fir_quant_norm(bufferA, &conv2_w, &conv2_b, bufferB, norm_l1, in_num); swap_buffer(bufferA, bufferB);
        #endif

        #ifdef norm_l2
            conv2d_fir_quant_norm_l2(bufferA, &conv2_w, &conv2_b, bufferB, norm_l1, in_num); swap_buffer(bufferA, bufferB);
        #endif
            
        #ifdef conv
            conv2d_fir(bufferA, &conv2_w, &conv2_b, bufferB); swap_buffer(bufferA, bufferB);
        #endif

        #ifdef drop
            conv2d_fir_dis(bufferA, &conv2_w, &conv2_b, bufferB, quant_list); swap_buffer(bufferA, bufferB);
        #endif

            quant(bufferA, bit_length);        
            maxpool(bufferA, bufferB, 2); swap_buffer(bufferA, bufferB);
            

            flatten(bufferA);
        
            fc(bufferA, &fc1_w, &fc1_b, bufferB); swap_buffer(bufferA, bufferB);
            
            quant(bufferA, bit_length);
            relu(bufferA, bufferB); swap_buffer(bufferA, bufferB);

            fc(bufferA, &fc2_w, &fc2_b, bufferB);

            /*************** prediction ******************/
            unsigned prediction = 0;
            
            //printf("%d'th quant %dth input:\n",tt, i);
            for(int k = 0; k < 10; ++k){
                //printf("%d: %d\n", k, bufferB->data[k]);
                result_list[tt][i].pred_list[k] = bufferB->data[k];
                
                if(bufferB->data[k] > bufferB->data[prediction]) prediction = k;
            }
            result_list[tt][i].pred = prediction;
            result_list[tt][i].ans = ans; 
            //printf("prediction: %d, ans = %d,",prediction, ans);
            
            
            if (prediction == ans) {
                right++; 
                result_list[tt][i].res = true;
                //printf(" result: true\n");
            } else {
                //printf(" result: false\n");  
            }
            
            //caculate confidence using core margin
            int first = 0, second = 0;
            //float entropy = 0;
            for (int k = 0; k < 10; ++k) {
                if (bufferB->data[k] > first && bufferB->data[k] > second) {
                    second = first;
                    first = bufferB->data[k];
                } else if (bufferB->data[k] > second){
                    second = bufferB->data[k];
                }
                //entropy += bufferB->data[k] * log(bufferB->data[k])
            }
            result_list[tt][i].core_margin = first - second;
            //printf("core margin = %d\n", result_list[tt][i].core_margin);

            // class_list[ans]++;
            // if (prediction == ans) {
            //     // right ans
            //     right++;  
            // } else {
            //     //wrong ans
            //     wrong_list[ans]++;

            // }
            
        }
        //printf("right cnt = %d\n",right);
        accuracy[tt] = right;
        printf("accuracy = %f\n", ((float)right)/in_num);




    #if defined(norm) || defined(norm_l2)
    /********* print norm l1 ***********/
        for (int i = 0; i < 40; ++i) {
            //printf("Norm l1_list[%d] = %lf\n",i, norm_l1[i]);
            printf("%lf\n",norm_l1[i]);
        }
        for (int i = 0; i < 40; ++i) {
        // printf("Norm l1_list[%d] = %lf\n",i, norm_l1[i]);
        printf("%lf,",norm_l1[i]);
        }
    #endif



    }
    fclose(fp);




    // partial right, full right
    printf("\npartial right, full right:\n");
    for (int i = 0; i < in_num; ++i) {
        if (result_list[0][i].res == true && result_list[1][i].res == true) {
            printf("%d ",i);
        }
    }
    // partial false, full right
    printf("\npartial false, full right:\n");
    for (int i = 0; i < in_num; ++i) {
        if (result_list[0][i].res == true && result_list[1][i].res == false) {
            printf("%d ",i);
        }
    }

    // printf("\nEnter sample number: (enter ctrl^z to exit)\n");
    // int sample_num;
    // while(scanf("%d",&sample_num) != -1) {
    //     printf("input sample %d\n",sample_num);
    //     printf("no drop:\n");
    //     printf("pred = %d, ans = %d, res = %d, core_margin = %d\n", result_list[0][sample_num].pred, result_list[0][sample_num].ans, result_list[0][sample_num].res, result_list[0][sample_num].core_margin);
    //     // for (int i = 0; i < 10; ++i) {
    //     //     printf("%d: %d\n", i, result_list[0][sample_num].pred_list[i]);
    //     // }
    //     printf("drop:\n");
    //     printf("pred = %d, ans = %d, res = %d, core_margin = %d\n", result_list[1][sample_num].pred, result_list[1][sample_num].ans, result_list[1][sample_num].res, result_list[1][sample_num].core_margin);
    //     // for (int i = 0; i < 10; ++i) {
    //     //     printf("%d: %d\n", i, result_list[1][sample_num].pred_list[i]);
    //     // }
    // }
    // printf("no drop core margin:\n");
    // for (int i = 0; i < in_num; ++i) {
    //     printf("%dth: %d ", i, result_list[0][i].core_margin);
    // }
    // printf("\n");
    // printf("drop core margin:\n");
    // for (int i = 0; i < in_num; ++i) {
    //     printf("%dth: %d ", i, result_list[1][i].core_margin);
    // }

    printf("\nEnter threshold: (enter ctrl^z to exit)\n");
    int threshold = 0;
    while(scanf("%d", &threshold) != -1) {
        int TT = 0, FT = 0, TF = 0, FF = 0;


        printf("those input's core_margin is below the threshold %d:\n", threshold);
        for (int i = 0; i < in_num; ++i) {
            if (result_list[1][i].core_margin <= threshold) {
                //printf("%d ", i);
                if (result_list[1][i].res == 0 && result_list[0][i].res == 1) { //抓出來了
                    TT ++;
                } else if (result_list[1][i].res == 1 && result_list[0][i].res == 1) { //多抓
                    TF ++;
                } else { 
                    FF ++; //只是用來統計有多少 input 的 confidence 是不夠的
                }
                next_level_list[next_level_idx++] = i; // 表達有那些 input 可以進到下一輪
            } else {
                if (result_list[1][i].res == 0 && result_list[0][i].res == 1) { //漏抓
                    FT ++;
                }
            }
        }
        printf("\ndiff between drop and non-drop = %d\n", accuracy[0] - accuracy[1]);
        printf("%d input's confidence is below threshold", TT + TF + FF);
        printf("\nTT = %d (%f), TF = %d (%f), FT = %d (%f)", TT, ((float)TT/(accuracy[0] - accuracy[1])), TF, ((float)TF/(accuracy[0] - accuracy[1])), FT, ((float)FT/(accuracy[0] - accuracy[1])));
        printf("\n\n");

    }

    printf("\nEnter sample number: (enter ctrl^z to exit)\n");
    int sample_num;
    while(scanf("%d",&sample_num) != -1) {
        printf("input sample %d\n",sample_num);
        printf("no drop:\n");
        printf("pred = %d, ans = %d, res = %d, core_margin = %d\n", result_list[0][sample_num].pred, result_list[0][sample_num].ans, result_list[0][sample_num].res, result_list[0][sample_num].core_margin);
        // for (int i = 0; i < 10; ++i) {
        //     printf("%d: %d\n", i, result_list[0][sample_num].pred_list[i]);
        // }
        printf("drop:\n");
        printf("pred = %d, ans = %d, res = %d, core_margin = %d\n", result_list[1][sample_num].pred, result_list[1][sample_num].ans, result_list[1][sample_num].res, result_list[1][sample_num].core_margin);
        // for (int i = 0; i < 10; ++i) {
        //     printf("%d: %d\n", i, result_list[1][sample_num].pred_list[i]);
        // }
    }
    printf("no drop core margin:\n");
    for (int i = 0; i < in_num; ++i) {
        printf("%dth: %d ", i, result_list[0][i].core_margin);
    }
    printf("\n");
    printf("drop core margin:\n");
    for (int i = 0; i < in_num; ++i) {
        printf("%dth: %d ", i, result_list[1][i].core_margin);
    }
    

        

    return 0;
}

//7 9 12 20 30 32 35 44 45 61 68 69 76 78 87 90 