#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 73
#define N_INPUT_2_1 70
#define N_OUT_2 128
#define N_OUT_2 128
#define N_LAYER_4 64
#define N_LAYER_4 64
#define N_INPUT_1_8 73
#define N_INPUT_2_8 64
#define N_TIME_STEPS_10 73
#define N_OUT_10 64
#define N_TIME_STEPS_10 73
#define N_OUT_10 64
#define N_OUTPUTS_17 73
#define N_FILT_17 4
#define N_LAYER_1_12 73
#define N_LAYER_2_12 4
#define N_OUTPUTS_18 73
#define N_FILT_18 70

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<55,16> accum2_t;
typedef ap_fixed<16,2> forward_weight2_t;
typedef ap_fixed<16,2> backward_weight2_t;
typedef ap_fixed<16,2> forward_recurrent_weight2_t;
typedef ap_fixed<16,2> backward_recurrent_weight2_t;
typedef ap_fixed<16,2> forward_bias2_t;
typedef ap_fixed<16,2> forward_recurrent_bias2_t;
typedef ap_fixed<16,2> backward_bias2_t;
typedef ap_fixed<16,2> backward_recurrent_bias2_t;
typedef ap_fixed<16,1,AP_RND_CONV,AP_SAT> act2_t;
typedef ap_ufixed<16,0,AP_RND_CONV,AP_SAT> recr_act2_t;
typedef ap_fixed<16,7,AP_RND_CONV,AP_SAT> state2_t;
typedef ap_ufixed<2,0> slope2_t;
typedef ap_ufixed<2,0> shift2_t;
typedef ap_fixed<33,8> layer2_t;
typedef ap_fixed<39,16> accum_dense2_t;
typedef ap_fixed<16,7,AP_RND_CONV,AP_SAT> layer3_t;
typedef ap_fixed<18,8> q_act_postencoder_table_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<16,2> weight4_t;
typedef ap_fixed<16,2> bias4_t;
typedef ap_uint<1> layer4_index;
typedef ap_fixed<16,7,AP_RND_CONV,AP_SAT> layer6_t;
typedef ap_fixed<18,8> q_act_dense_mean_table_t;
typedef ap_fixed<16,6> input8_t;
typedef ap_fixed<54,15> accum10_t;
typedef ap_fixed<16,2> weight10_t;
typedef ap_fixed<16,2> recurrent_weight10_t;
typedef ap_fixed<16,2> bias10_t;
typedef ap_fixed<16,2> recurrent_bias10_t;
typedef ap_fixed<16,1,AP_RND_CONV,AP_SAT> act10_t;
typedef ap_ufixed<16,0,AP_RND_CONV,AP_SAT> recr_act10_t;
typedef ap_fixed<16,7,AP_RND_CONV,AP_SAT> state10_t;
typedef ap_ufixed<2,0> slope10_t;
typedef ap_ufixed<2,0> shift10_t;
typedef ap_fixed<33,8> layer10_t;
typedef ap_fixed<38,15> accum_dense10_t;
typedef ap_fixed<16,7,AP_RND_CONV,AP_SAT> layer11_t;
typedef ap_fixed<18,8> q_act_postdecoder_table_t;
typedef ap_fixed<16,6> layer17_t;
typedef ap_fixed<16,2> dense_weight_t;
typedef ap_uint<1> bias17_t;
typedef ap_fixed<16,7,AP_RND_CONV,AP_SAT> layer14_t;
typedef ap_fixed<18,8> q_act_postdense_table_t;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<16,2> neuraldense_weight_t;
typedef ap_fixed<16,2> neuraldense_bias_t;
typedef ap_fixed<22,6> dense_accum_t;

#endif
