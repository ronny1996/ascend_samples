#include "npu_runner.h"

int main(int argc, char const *argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    std::vector<int64_t> x_shape({2, 2, 2, 2});
    std::vector<float> x_data, y_grad_data;
    for (auto i = 0; i < CommonHelper::Product(x_shape); i++) {
      x_data.push_back(rand() % 10000 * 1.0 / 10000 - 0.5);
      y_grad_data.push_back(rand() % 10000 * 1.0 / 10000 - 0.5);
    }
    NpuHelper::Profiler profiler("/work/npu_prof/");
    {
      NpuTensor<float> x_tensor({x_shape}, x_data);


      NpuTensor<float> scale_tensor({x_shape[1]}, std::vector<float>(x_shape[1], rand() % 10000 / 10000 - 0.5));
      NpuTensor<float> offset_tensor({x_shape[1]}, std::vector<float>(x_shape[1], rand() % 10000 / 10000 - 0.5));
      NpuTensor<float> mean_tensor({x_shape[1]}, std::vector<float>(x_shape[1], rand() % 10000 / 10000 - 0.5));
      NpuTensor<float> variance_tensor({x_shape[1]}, std::vector<float>(x_shape[1], rand() % 10000 / 10000 - 0.5));

      NpuTensor<float> out_tensor({x_shape});
      NpuTensor<float> batch_mean({x_shape[1]}, std::vector<float>(x_shape[1], rand() % 10000 / 10000 - 0.5));
      NpuTensor<float> batch_variance({x_shape[1]}, std::vector<float>(x_shape[1], rand() % 10000 / 10000 - 0.5));
      NpuTensor<float> reserve_space_1({x_shape[1]}, std::vector<float>(x_shape[1], rand() % 10000 / 10000 - 0.5));
      NpuTensor<float> reserve_space_2({x_shape[1]}, std::vector<float>(x_shape[1], rand() % 10000 / 10000 - 0.5));

      {
        NpuRunner runner("BatchNorm");
        runner.AddInput(x_tensor)
            .AddInput(scale_tensor)
            .AddInput(offset_tensor)
            // .AddInput(mean_tensor)  // Must be"None" if the operation is used for
            //                         // training
            // .AddInput(variance_tensor)  // Must be"None" if the operation is used
            //                             // for training
            .AddOutput(out_tensor)
            .AddOutput(reserve_space_1)
            .AddOutput(reserve_space_2)
            .AddOutput(batch_mean)  // mean_for_grad
            .AddOutput(batch_variance)  // variance_for_grad
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
      {
        NpuRunner runner("BatchNorm");
        runner.AddInput(x_tensor)
            .AddInput(scale_tensor)
            .AddInput(offset_tensor)
            .AddInput(batch_mean)  // Must be"None" if the operation is used for
                                    // training
            .AddInput(batch_variance)  // Must be"None" if the operation is used
                                        // for training
            .AddOutput(out_tensor)
            .AddOutput(reserve_space_1)
            .AddOutput(reserve_space_2)
            .AddOutput(reserve_space_1)  // mean_for_grad
            .AddOutput(reserve_space_2)  // variance_for_grad
            .SetAttr("epsilon", static_cast<float>(1e-4))
            .SetAttr("data_format", "NCHW")
            .SetAttr("is_training", false)
            .Run();
      }
      out_tensor.print();
      batch_mean.print();
      batch_variance.print();
      reserve_space_1.print();
      reserve_space_2.print();
    }
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
