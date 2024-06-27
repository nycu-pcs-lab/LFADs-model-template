#ifndef MYPROJECT_BRIDGE_H_
#define MYPROJECT_BRIDGE_H_

#include "firmware/myproject.h"
#include "firmware/nnet_utils/nnet_helpers.h"
#include <algorithm>
#include <map>

// hls-fpga-machine-learning insert bram

namespace nnet {
bool trace_enabled = false;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

extern "C" {

struct trace_data {
    const char *name;
    void *data;
};

void allocate_trace_storage(size_t element_size) {
    nnet::trace_enabled = true;
    nnet::trace_outputs = new std::map<std::string, void *>;
    nnet::trace_type_size = element_size;
}

void free_trace_storage() {
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        void *ptr = i->second;
        free(ptr);
    }
    nnet::trace_outputs->clear();
    delete nnet::trace_outputs;
    nnet::trace_outputs = NULL;
    nnet::trace_enabled = false;
}

void collect_trace_output(struct trace_data *c_trace_outputs) {
    int ii = 0;
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        c_trace_outputs[ii].name = i->first.c_str();
        c_trace_outputs[ii].data = i->second;
        ii++;
    }
}

// Wrapper of top level function for Python bridge
void myproject_float(
    float encoder_input[N_INPUT_1_1*N_INPUT_2_1], float decoder_input[N_INPUT_1_8*N_INPUT_2_8],
    float layer18_out[N_OUTPUTS_18*N_FILT_18]
) {

    hls::stream<input_t> encoder_input_ap[70];
    nnet::convert_data<float, input_t, 70, N_INPUT_1_1*N_INPUT_2_1/70>(encoder_input, encoder_input_ap);
    hls::stream<input8_t> decoder_input_ap[64];
    nnet::convert_data<float, input8_t, 64, N_INPUT_1_8*N_INPUT_2_8/64>(decoder_input, decoder_input_ap);

    hls::stream<result_t> layer18_out_ap[70];

    myproject(encoder_input_ap,decoder_input_ap,layer18_out_ap);

    nnet::convert_data<result_t, float, 70, N_OUTPUTS_18*N_FILT_18/70>(layer18_out_ap, layer18_out);
}

void myproject_double(
    double encoder_input[N_INPUT_1_1*N_INPUT_2_1], double decoder_input[N_INPUT_1_8*N_INPUT_2_8],
    double layer18_out[N_OUTPUTS_18*N_FILT_18]
) {
    hls::stream<input_t> encoder_input_ap[70];
    nnet::convert_data<double, input_t, 70, N_INPUT_1_1*N_INPUT_2_1/70>(encoder_input, encoder_input_ap);
    hls::stream<input8_t> decoder_input_ap[64];
    nnet::convert_data<double, input8_t, 64, N_INPUT_1_8*N_INPUT_2_8/64>(decoder_input, decoder_input_ap);

    hls::stream<result_t> layer18_out_ap[70];

    myproject(encoder_input_ap,decoder_input_ap,layer18_out_ap);

    nnet::convert_data<result_t, double, 70, N_OUTPUTS_18*N_FILT_18/70>(layer18_out_ap, layer18_out);
}
}

#endif
