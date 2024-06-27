#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    hls::stream<input_t> encoder_input[70], hls::stream<input8_t> decoder_input[64],
    hls::stream<result_t> layer18_out[70]
);

#endif
