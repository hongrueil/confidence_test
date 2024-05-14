#include "../../inference/nn.h"

//#include "weight.h"
#include "q15_para.h"

const Tensor conv1_w = {
    .ndim = 4,
    .dims = {20, 1, 5, 5},
    .bw = 0,
    .data = conv1_w_raw
};

const Tensor conv1_b = {
    .ndim = 1,
    .dims = {20},
    .bw = 0,
    .data = conv1_b_raw
};

const Tensor conv2_w = {
    .ndim = 4,
    .dims = {40, 20, 5, 5},
    .bw = 0,
    .data = conv2_w_raw
};

const Tensor conv2_b = {
    .ndim = 1,
    .dims = {40},
    .bw = 0,
    .data = conv2_b_raw
};

const Tensor fc1_w = {
    .ndim = 2,
    .dims = {64, 640},
    .bw = 0,
    .data = fc1_w_raw
};

const Tensor fc1_b = {
    .ndim = 1,
    .dims = {64},
    .bw = 0,
    .data = fc1_b_raw
};

const Tensor fc2_w = {
    .ndim = 2,
    .dims = {10, 64},
    .bw = 0,
    .data = fc2_w_raw
};

const Tensor fc2_b = {
    .ndim = 1,
    .dims = {10},
    .bw = 0,
    .data = fc2_b_raw
};
