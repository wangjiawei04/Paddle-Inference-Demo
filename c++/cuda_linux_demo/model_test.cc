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
  config.SetModel("bert_seq128_model"); // Load no-combined model
  config.EnableUseGpu(100, 0);
  config.SwitchIrOptim(true);
  config.EnableMemoryOptim();
  
  // Create predictor
  auto predictor = paddle_infer::CreatePredictor(config);


  std::vector<std::string> input_names = {"input_ids", "position_ids", "segment_ids", "input_mask"};
  std::vector<int> input_shape = {1,128,1};
  std::vector<int64_t> input0(128,0);
  std::vector<int64_t> input1(128,0);
  std::vector<int64_t> input2(128,0);
  std::vector<float> mask(128,0.0);
  
  std::vector<int64_t> tmp = {101, 6843, 3241, 749, 8024, 7662, 2533, 1391,2533, 2523, 7676};
  for (size_t i = 0; i< 11; i++) {
    input0[i] = tmp[i];
    input1[i] = i;
    input2[i] = 0;
    mask[i] = 1.0;
  }

  auto input_t0 = predictor->GetInputHandle(input_names[0]);
  input_t0->Reshape(input_shape);
  input_t0->CopyFromCpu(input0.data());
  auto input_t1 = predictor->GetInputHandle(input_names[1]);
  input_t1->Reshape(input_shape);
  input_t1->CopyFromCpu(input1.data());
  auto input_t2 = predictor->GetInputHandle(input_names[2]);
  input_t2->Reshape(input_shape);
  input_t2->CopyFromCpu(input2.data());
  auto input_t3 = predictor->GetInputHandle(input_names[3]);
  input_t3->Reshape(input_shape);
  input_t3->CopyFromCpu(mask.data());

  // Run
  predictor->Run();

  // Get output
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
  }
  return 0;
}
