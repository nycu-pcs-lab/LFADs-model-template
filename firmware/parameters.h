#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_array_stream.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_bidirectional.h"
#include "nnet_utils/nnet_conv1d.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_array_stream.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_recurrent.h"
#include "nnet_utils/nnet_recurrent_array_stream.h"
#include "nnet_utils/nnet_sepconv1d_array_stream.h"
#include "nnet_utils/nnet_sepconv1d_stream.h"

// hls-fpga-machine-learning insert weights
#include "weights/fw2.h"
#include "weights/bw2.h"
#include "weights/fwr2.h"
#include "weights/bwr2.h"
#include "weights/fb2.h"
#include "weights/fbr2.h"
#include "weights/bb2.h"
#include "weights/bbr2.h"
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/w10.h"
#include "weights/wr10.h"
#include "weights/b10.h"
#include "weights/br10.h"
#include "weights/w17.h"
#include "weights/b17.h"
#include "weights/w18.h"
#include "weights/b18.h"

// hls-fpga-machine-learning insert layer-config
// EncoderRNN
struct config2_1 : nnet::dense_config {
    static const unsigned n_in = N_INPUT_2_1;
    static const unsigned n_out = N_OUT_2 * 3 /2;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = N_INPUT_2_1;
    static const unsigned n_zeros = 1;
    static const unsigned n_nonzeros = 13439;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef accum_dense2_t accum_t;
    typedef forward_bias2_t bias_t;
    typedef forward_weight2_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config2_2 : nnet::dense_config {
    static const unsigned n_in = N_OUT_2/2;
    static const unsigned n_out = N_OUT_2 * 3 /2;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = N_OUT_2/2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 12288;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef accum_dense2_t accum_t;
    typedef forward_bias2_t bias_t;
    typedef forward_weight2_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct hard_sigmoid_config2_recr  : nnet::hard_activ_config {
    static const unsigned n_in = N_OUT_2 * 2 /2;
    static const slope2_t slope;
    static const shift2_t shift;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned reuse_factor = 1;
};
const slope2_t hard_sigmoid_config2_recr::slope = 0.5;
const shift2_t hard_sigmoid_config2_recr::shift = 0.5;

struct hard_tanh_config2 : nnet::hard_activ_config{
    static const unsigned n_in = N_OUT_2/2;
    static const slope2_t slope;
    static const shift2_t shift;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned reuse_factor = 1;
};
const slope2_t hard_tanh_config2::slope = 0.5;
const shift2_t hard_tanh_config2::shift = 0.5;

struct config2_f : nnet::gru_config {
    typedef accum_dense2_t accum_dense_t;
    typedef accum2_t accum_t;
    typedef forward_weight2_t weight_t;  // Matrix
    typedef forward_bias2_t bias_t;  // Vector
    typedef config2_1 mult_config1;
    typedef config2_2 mult_config2;
    typedef hard_sigmoid_config2_recr ACT_CONFIG_GRU;
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::hard_sigmoid<x_T, y_T, config_T>;
    typedef hard_tanh_config2 ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::hard_tanh<x_T, y_T, config_T>;
    static const unsigned n_in  = N_INPUT_2_1;
    static const unsigned n_out = 64;
    static const unsigned n_state = 64;
    static const unsigned n_sequence = N_INPUT_1_1;
    static const unsigned n_sequence_out = 1;
    static const unsigned io_type = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;
    typedef state2_t state_t;
    typedef act2_t act_t;
    typedef recr_act2_t recr_act_t;
    static const unsigned merge_mode = nnet::concat;
};

struct config2_b : nnet::gru_config {
    typedef accum_dense2_t accum_dense_t;
    typedef accum2_t accum_t;
    typedef backward_weight2_t weight_t;  // Matrix
    typedef backward_bias2_t bias_t;  // Vector
    typedef config2_1 mult_config1;
    typedef config2_2 mult_config2;
    typedef hard_sigmoid_config2_recr ACT_CONFIG_GRU;
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::hard_sigmoid<x_T, y_T, config_T>;
    typedef hard_tanh_config2 ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::hard_tanh<x_T, y_T, config_T>;
    static const unsigned n_in  = N_INPUT_2_1;
    static const unsigned n_out = 64;
    static const unsigned n_state = 64;
    static const unsigned n_sequence = N_INPUT_1_1;
    static const unsigned n_sequence_out = 1;
    static const unsigned io_type = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;
    typedef state2_t state_t;
    typedef act2_t act_t;
    typedef recr_act2_t recr_act_t;
    static const unsigned merge_mode = nnet::concat;
};

struct config2 : nnet::bidirectional_config {
    typedef accum_dense2_t accum_dense_t;
    typedef accum2_t accum_t;
    typedef backward_weight2_t weight_t;  // Matrix
    typedef backward_bias2_t bias_t;  // Vector

    typedef config2_f config_rnn_layer_f;
    typedef config2_b config_rnn_layer_b;
    static const unsigned n_in  = N_INPUT_2_1;
    static const unsigned n_out = N_OUT_2;
    static const unsigned n_state = N_OUT_2/2;
    static const unsigned n_sequence = N_INPUT_1_1;
    static const unsigned n_sequence_out = 1;
    static const unsigned io_type = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned merge_mode = nnet::concat;
};

// q_act_postencoder
struct linear_config3 : nnet::activ_config {
    static const unsigned n_in = 128;
    static const unsigned n_chan = N_OUT_2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned reuse_factor = 1;
    typedef q_act_postencoder_table_t table_t;
};

// DenseMean
struct config4 : nnet::dense_config {
    static const unsigned n_in = 128;
    static const unsigned n_out = 64;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 128;
    static const unsigned n_zeros = 2;
    static const unsigned n_nonzeros = 8190;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef dense_accum_t accum_t;
    typedef bias4_t bias_t;
    typedef weight4_t weight_t;
    typedef layer4_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_act_dense_mean
struct linear_config6 : nnet::activ_config {
    static const unsigned n_in = 64;
    static const unsigned n_chan = N_LAYER_4;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned reuse_factor = 1;
    typedef q_act_dense_mean_table_t table_t;
};

// DecoderGRU
struct config10_1 : nnet::dense_config {
    static const unsigned n_in = N_INPUT_2_8;
    static const unsigned n_out = N_OUT_10 * 3;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = N_INPUT_2_8;
    static const unsigned n_zeros = 3;
    static const unsigned n_nonzeros = 12285;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef accum_dense10_t accum_t;
    typedef bias10_t bias_t;
    typedef weight10_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config10_2 : nnet::dense_config {
    static const unsigned n_in = N_OUT_10;
    static const unsigned n_out = N_OUT_10 * 3;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = N_OUT_10;
    static const unsigned n_zeros = 3044;
    static const unsigned n_nonzeros = 9244;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef accum_dense10_t accum_t;
    typedef bias10_t bias_t;
    typedef weight10_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct hard_sigmoid_config10_recr  : nnet::hard_activ_config {
    static const unsigned n_in = N_OUT_10 * 2;
    static const slope10_t slope;
    static const shift10_t shift;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned reuse_factor = 1;
};
const slope10_t hard_sigmoid_config10_recr::slope = 0.5;
const shift10_t hard_sigmoid_config10_recr::shift = 0.5;

struct hard_tanh_config10 : nnet::hard_activ_config{
    static const unsigned n_in = N_OUT_10;
    static const slope10_t slope;
    static const shift10_t shift;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned reuse_factor = 1;
};
const slope10_t hard_tanh_config10::slope = 0.5;
const shift10_t hard_tanh_config10::shift = 0.5;

struct config10 : nnet::gru_config {
    typedef accum_dense10_t accum_dense_t;
    typedef accum10_t accum_t;
    typedef weight10_t weight_t;  // Matrix
    typedef bias10_t bias_t;  // Vector
    typedef config10_1 mult_config1;
    typedef config10_2 mult_config2;
    typedef hard_sigmoid_config10_recr ACT_CONFIG_GRU;
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::hard_sigmoid<x_T, y_T, config_T>;
    typedef hard_tanh_config10 ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::hard_tanh<x_T, y_T, config_T>;
    static const unsigned n_in  = N_INPUT_2_8;
    static const unsigned n_out = N_OUT_10;
    static const unsigned n_state = N_OUT_10;
    static const unsigned n_sequence = N_INPUT_1_8;
    static const unsigned n_sequence_out = N_TIME_STEPS_10;
    static const unsigned io_type = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;
    static const bool use_initial = 1;
    typedef state10_t state_t;
    typedef act10_t act_t;
    typedef recr_act10_t recr_act_t;
};

// q_act_postdecoder
struct linear_config11 : nnet::activ_config {
    static const unsigned n_in = 4672;
    static const unsigned n_chan = N_OUT_10;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned reuse_factor = 1;
    typedef q_act_postdecoder_table_t table_t;
};

// Dense
struct config17_mult : nnet::dense_config {
    static const unsigned n_in = 64;
    static const unsigned n_out = 4;
    static const unsigned reuse_factor = 64;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef dense_accum_t accum_t;
    typedef bias17_t bias_t;
    typedef dense_weight_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config17 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 73;
    static const unsigned n_chan = 64;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 4;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 73;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 73;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 73;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef bias17_t bias_t;
    typedef dense_weight_t weight_t;
    typedef config17_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_unscaled<K, S, W>;
};
const ap_uint<config17::filt_width> config17::pixels[] = {0};

// q_act_postdense
struct linear_config14 : nnet::activ_config {
    static const unsigned n_in = 292;
    static const unsigned n_chan = N_FILT_17;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_array_stream;
    static const unsigned reuse_factor = 1;
    typedef q_act_postdense_table_t table_t;
};

// NeuralDense
struct config18_mult : nnet::dense_config {
    static const unsigned n_in = 4;
    static const unsigned n_out = 70;
    static const unsigned reuse_factor = 4;
    static const unsigned strategy = nnet::resource;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef dense_accum_t accum_t;
    typedef neuraldense_bias_t bias_t;
    typedef neuraldense_weight_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config18 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 73;
    static const unsigned n_chan = 4;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 70;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 73;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 73;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 73;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef neuraldense_bias_t bias_t;
    typedef neuraldense_weight_t weight_t;
    typedef config18_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_unscaled<K, S, W>;
};
const ap_uint<config18::filt_width> config18::pixels[] = {0};


#endif
