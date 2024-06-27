#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> encoder_input[70], hls::stream<input8_t> decoder_input[64],
    hls::stream<result_t> layer18_out[70]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=encoder_input,decoder_input,layer18_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<forward_weight2_t, 13440>(fw2, "fw2.txt");
        nnet::load_weights_from_txt<backward_weight2_t, 13440>(bw2, "bw2.txt");
        nnet::load_weights_from_txt<forward_recurrent_weight2_t, 12288>(fwr2, "fwr2.txt");
        nnet::load_weights_from_txt<backward_recurrent_weight2_t, 12288>(bwr2, "bwr2.txt");
        nnet::load_weights_from_txt<forward_bias2_t, 192>(fb2, "fb2.txt");
        nnet::load_weights_from_txt<forward_recurrent_bias2_t, 192>(fbr2, "fbr2.txt");
        nnet::load_weights_from_txt<backward_bias2_t, 192>(bb2, "bb2.txt");
        nnet::load_weights_from_txt<backward_recurrent_bias2_t, 192>(bbr2, "bbr2.txt");
        nnet::load_weights_from_txt<weight4_t, 8192>(w4, "w4.txt");
        nnet::load_weights_from_txt<bias4_t, 64>(b4, "b4.txt");
        nnet::load_weights_from_txt<weight10_t, 12288>(w10, "w10.txt");
        nnet::load_weights_from_txt<recurrent_weight10_t, 12288>(wr10, "wr10.txt");
        nnet::load_weights_from_txt<bias10_t, 192>(b10, "b10.txt");
        nnet::load_weights_from_txt<recurrent_bias10_t, 192>(br10, "br10.txt");
        nnet::load_weights_from_txt<dense_weight_t, 256>(w17, "w17.txt");
        nnet::load_weights_from_txt<bias17_t, 4>(b17, "b17.txt");
        nnet::load_weights_from_txt<neuraldense_weight_t, 280>(w18, "w18.txt");
        nnet::load_weights_from_txt<neuraldense_bias_t, 70>(b18, "b18.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer2_t> layer2_out[128];
    #pragma HLS STREAM variable=layer2_out depth=1
    nnet::bidirectional_array<input_t, layer2_t, config2>(encoder_input, layer2_out, bw2, bwr2, bb2, bbr2, fw2, fwr2, fb2, fbr2); // EncoderRNN

    hls::stream<layer3_t> layer3_out[128];
    #pragma HLS STREAM variable=layer3_out depth=1
    nnet::linear<layer2_t, layer3_t, linear_config3>(layer2_out, layer3_out); // q_act_postencoder

    hls::stream<layer4_t> layer4_out[64];
    #pragma HLS STREAM variable=layer4_out depth=1
    nnet::dense<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, b4); // DenseMean

    hls::stream<layer6_t> layer6_out[64];
    #pragma HLS STREAM variable=layer6_out depth=1
    nnet::linear<layer4_t, layer6_t, linear_config6>(layer4_out, layer6_out); // q_act_dense_mean

    hls::stream<layer10_t> layer10_out[64];
    #pragma HLS STREAM variable=layer10_out depth=73
    nnet::gru_stack<input8_t, layer6_t, layer10_t, config10>(decoder_input, layer6_out, layer10_out, w10, wr10, b10, br10); // DecoderGRU

    hls::stream<layer11_t> layer11_out[64];
    #pragma HLS STREAM variable=layer11_out depth=73
    nnet::linear<layer10_t, layer11_t, linear_config11>(layer10_out, layer11_out); // q_act_postdecoder

    hls::stream<layer17_t> layer17_out[4];
    #pragma HLS STREAM variable=layer17_out depth=73
    nnet::pointwise_conv_1d_cl<layer11_t, layer17_t, config17>(layer11_out, layer17_out, w17, b17); // Dense

    hls::stream<layer14_t> layer14_out[4];
    #pragma HLS STREAM variable=layer14_out depth=73
    nnet::linear<layer17_t, layer14_t, linear_config14>(layer17_out, layer14_out); // q_act_postdense

    nnet::pointwise_conv_1d_cl<layer14_t, result_t, config18>(layer14_out, layer18_out, w18, b18); // NeuralDense

}
