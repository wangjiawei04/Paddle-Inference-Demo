#include <assert.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>

#include "paddle_inference_api.h"

int main(int argc, char *argv[]) {

  // Init config
  paddle_infer::Config config;
  config.SetModel("cascade_model"); // Load no-combined model
  config.EnableUseGpu(100, 2);
  config.SwitchIrOptim(true);
  config.EnableMemoryOptim();
  config.SwitchIrOptim(true); 
  // Create predictor
  auto predictor = paddle_infer::CreatePredictor(config); 
  // ((1, 3, 800, 1088), (1, 3), (1, 3))
  // im info /shape 800,1088,1
  std::unordered_map<std::string, std::vector<float> > feed_values;
  std::unordered_map<std::string, std::vector<int> > feed_shape;
  std::vector<std::string> input_names = {"image", "im_info", "im_shape"};

  feed_values[input_names[0]] = std::vector<float>(1*3*800*1088, 1.0); //default 1.0
  feed_shape[input_names[0]] = {1,3,800,1088};
  feed_values[input_names[1]] = {800, 1088, 1};
  feed_shape[input_names[1]] = {1,3};
  feed_values[input_names[2]] = {800, 1088, 1};
  feed_shape[input_names[2]] = {1,3};
  
  for (int i = 0; i < input_names.size(); ++i) {
    auto input=predictor->GetInputHandle(input_names[i]);
    input->Reshape(feed_shape[input_names[i]]);
    input->CopyFromCpu(feed_values[input_names[i]].data());
  }

  // Run
  predictor->Run();

  // Get output
  /*
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());
  std::vector<float> out_data;
  out_data.resize(out_num);
  output_t->CopyToCpu(out_data.data());
  for (size_t i = 0; i < out_data.size(); i++) {
    std::cout << "output[" << i << "]: " << out_data[i] << std::endl;
  } */
  return 0;
}
