#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  {
    NpuTensor<float> x_tensor({2, 1, 2, 2},
                              {1, 1, 1, 1, 2, 2, 2, 2});
    NpuTensor<float> scale_tensor({1}, {1});
    NpuTensor<float> offset_tensor({1}, {0});
    NpuTensor<float> mean_tensor({1}, {1});
    NpuTensor<float> variance_tensor({1}, {1});

    NpuTensor<float> out_tensor({2, 1, 2, 2});
    NpuTensor<float> batch_mean({1}, {1});
    NpuTensor<float> batch_variance({1}, {1});
    NpuTensor<float> reserve_space_1({1}, {1});
    NpuTensor<float> reserve_space_2({1}, {1});
    {
      NpuRunner runner("BatchNorm");
      runner.AddInput(x_tensor)
          .AddInput(scale_tensor)
          .AddInput(offset_tensor)
          .AddInput(mean_tensor) // Must be"None" if the operation is used for training
          .AddInput(variance_tensor) // Must be"None" if the operation is used for training
          .AddOutput(out_tensor)
          .AddOutput(batch_mean)
          .AddOutput(batch_variance)
          .AddOutput(reserve_space_1) // mean_for_grad
          .AddOutput(reserve_space_2) // variance_for_grad
          .SetAttr("epsilon", static_cast<float>(1e-4))
          .SetAttr("data_format", "NCHW")
          .SetAttr("is_training", true)
          .Run();
    }
    out_tensor.print();
    batch_mean.print();
    batch_variance.print();
    reserve_space_1.print();
    reserve_space_2.print();

    NpuTensor<float> y_grad_tensor({2, 1, 2, 2}, std::vector<float>(8, 1.0f));
    NpuTensor<float> x_grad_tensor({2, 1, 2, 2}, std::vector<float>(8, 0.0f));
    NpuTensor<float> scale_grad_tensor({1}, std::vector<float>(1, 0.0f));
    NpuTensor<float> offset_grad_tensor({1}, std::vector<float>(1, 0.0f));
    NpuTensor<float> reserve_space_3({1}, std::vector<float>(1, 0.0f));
    NpuTensor<float> reserve_space_4({1}, std::vector<float>(1, 0.0f));
    {
      NpuRunner runner("BatchNormGrad");
      runner.AddInput(y_grad_tensor)
      .AddInput(x_tensor)
      .AddInput(scale_tensor)
      .AddInput(reserve_space_1)
      .AddInput(reserve_space_2)
      .AddOutput(x_grad_tensor)
      .AddOutput(scale_grad_tensor)
      .AddOutput(offset_grad_tensor)
      .SetAttr("data_format", "NCHW")
      .Run();
    }
    x_grad_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
