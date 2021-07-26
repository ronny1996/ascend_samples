#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    std::vector<int64_t> x_shape({2, 3, 4, 8});
    std::vector<int64_t> x_shape_truncated({x_shape[2], x_shape[3]});
    std::vector<int64_t> out_shape({2, 3, 2, 4});
    std::vector<int64_t> out_shape_truncated({out_shape[2], out_shape[3]});

    size_t x_numel = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<int64_t>());
    size_t out_numel = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int64_t>());
    NpuTensor<float> x_tensor(x_shape, std::vector<float>(x_numel, 1.0f));
    NpuTensor<float> out_tensor(out_shape);
    { // test AdaptiveAvgPool2d
      NpuRunner runner("AdaptiveAvgPool2d");
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
          .SetAttr("output_size", out_shape_truncated)
          .Run();
    }
    out_tensor.print();
    
    NpuTensor<float> x_grad_tensor(x_shape);
    NpuTensor<float> out_grad_tensor(out_shape,  std::vector<float>(out_numel, 1.0f));
    { // test AdaptiveAvgPool2dGrad
      NpuRunner runner("AdaptiveAvgPool2dGrad");
      runner.AddInput(out_grad_tensor)
          .AddOutput(x_grad_tensor)
          .SetAttr("orig_input_shape", x_shape)
          .Run();
    }
    x_grad_tensor.print();

    { // test AdaptiveMaxPool2d
      NpuRunner runner("AdaptiveMaxPool2d");
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
          .SetAttr("output_size", out_shape_truncated)
          .Run();
    }
    out_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
