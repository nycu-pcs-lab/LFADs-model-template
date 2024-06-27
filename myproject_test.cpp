#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "firmware/myproject.h"
#include "firmware/nnet_utils/nnet_helpers.h"

// hls-fpga-machine-learning insert bram

#define CHECKPOINT 5000

namespace nnet {
bool trace_enabled = true;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

int main(int argc, char **argv) {
    // load input data from text file
    std::ifstream fin("tb_data/tb_input_features.dat");
    // load predictions from text file
    std::ifstream fpr("tb_data/tb_output_predictions.dat");

#ifdef RTL_SIM
    std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
    std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
    std::ofstream fout(RESULTS_LOG);

    std::string iline;
    std::string pline;
    int e = 0;

    if (fin.is_open() && fpr.is_open()) {
        while (std::getline(fin, iline) && std::getline(fpr, pline)) {
            if (e % CHECKPOINT == 0)
                std::cout << "Processing input " << e << std::endl;
            char *cstr = const_cast<char *>(iline.c_str());
            char *current;
            std::vector<float> in;
            current = strtok(cstr, " ");
            while (current != NULL) {
                in.push_back(atof(current));
                current = strtok(NULL, " ");
            }
            cstr = const_cast<char *>(pline.c_str());
            std::vector<float> pr;
            current = strtok(cstr, " ");
            while (current != NULL) {
                pr.push_back(atof(current));
                current = strtok(NULL, " ");
            }

            // hls-fpga-machine-learning insert data
      hls::stream<input_t> encoder_input[70];
      nnet::copy_data_switch<float, input_t, 0, N_INPUT_1_1*N_INPUT_2_1, 70>(in, encoder_input);
      hls::stream<input8_t> decoder_input[64];
      nnet::copy_data_switch<float, input8_t, 5110, N_INPUT_1_8*N_INPUT_2_8, 64>(in, decoder_input);
      hls::stream<result_t> layer18_out[70];

            // hls-fpga-machine-learning insert top-level-function
            myproject(encoder_input,decoder_input,layer18_out);

            if (e % CHECKPOINT == 0) {
                std::cout << "Predictions" << std::endl;
                // hls-fpga-machine-learning insert predictions
                //for(int i = 0; i < N_OUTPUTS_18*N_FILT_18; i++) {
                //  std::cout << pr[i] << " ";
                //}
                std::cout << std::endl;
                std::cout << "Quantized predictions" << std::endl;
                // hls-fpga-machine-learning insert quantized
                // nnet::print_result<result_t, N_OUTPUTS_18*N_FILT_18, 70>(layer18_out, std::cout, true);
            }
            e++;

            // hls-fpga-machine-learning insert tb-output
            nnet::print_result_switch<result_t, N_OUTPUTS_18*N_FILT_18, 70>(layer18_out, fout);
        }
        fin.close();
        fpr.close();
    } else {
        std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;

        // hls-fpga-machine-learning insert zero
    hls::stream<input_t> encoder_input[70];
    nnet::fill_zero_switch<input_t, N_INPUT_1_1*N_INPUT_2_1, 70>(encoder_input);
    hls::stream<input8_t> decoder_input[64];
    nnet::fill_zero_switch<input8_t, N_INPUT_1_8*N_INPUT_2_8, 64>(decoder_input);
    hls::stream<result_t> layer18_out[70];

        // hls-fpga-machine-learning insert top-level-function
        myproject(encoder_input,decoder_input,layer18_out);

        // hls-fpga-machine-learning insert output
        nnet::print_result_switch<result_t, N_OUTPUTS_18*N_FILT_18, 70>(layer18_out, std::cout, true);

        // hls-fpga-machine-learning insert tb-output
        nnet::print_result_switch<result_t, N_OUTPUTS_18*N_FILT_18, 70>(layer18_out, fout);
    }

    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;
}
