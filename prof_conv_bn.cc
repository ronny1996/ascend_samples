#include "npu_runner.h"

#include <algorithm>

int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuHelper::Profiler profiler("/work/npu_prof/");
    {
      int groups = 1;
      std::vector<int64_t> x_shape({32, 3, 112, 112});
      std::vector<int64_t> out_shape({32, 32, 56, 56});
      int Cin = x_shape[1];
      int Cout = out_shape[1];
      std::vector<int64_t> filter_shape({Cout, Cin / groups, 2, 2});

      size_t x_numel = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<int64_t>());
      size_t filter_numel = std::accumulate(filter_shape.begin(), filter_shape.end(), 1, std::multiplies<int64_t>());
      size_t out_numel = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int64_t>());

      NpuTensor<npu::float16> x_tensor(x_shape);
      NpuTensor<npu::float16> filter_tensor(filter_shape); //ACL_FORMAT_FRACTAL_Z);
      NpuTensor<npu::float16> bias_tensor({1});
      NpuTensor<npu::float16> out_tensor(out_shape);
#if 1
      {
        NpuRunner runner("Conv2D");
        runner.AddInput(x_tensor)
            .AddInput(filter_tensor)
            .AddOutput(out_tensor)
            .SetAttr("strides", std::vector<int64_t>({1, 1, 2, 2})) // dataformat is same with input, n and c must be set to 1.
            .SetAttr("pads", std::vector<int64_t>({0, 0, 0, 0})) // t, b, l, r
            .SetAttr("dilations", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input
            .SetAttr("groups", static_cast<int64_t>(groups))
            .SetAttr("data_format", "NCHW")
            .Run();
      }
      {
        NpuRunner runner("Conv2D");
        runner.AddInput(x_tensor)
            .AddInput(filter_tensor)
            .AddOutput(out_tensor)
            .SetAttr("strides", std::vector<int64_t>({1, 1, 2, 2})) // dataformat is same with input, n and c must be set to 1.
            .SetAttr("pads", std::vector<int64_t>({0, 0, 0, 0})) // t, b, l, r
            .SetAttr("dilations", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input
            .SetAttr("groups", static_cast<int64_t>(groups))
            .SetAttr("data_format", "NCHW")
            .Run();
      }
#endif
#if 1
      NpuTensor<npu::float16> y_grad_tensor(out_shape);
      NpuTensor<npu::float16> x_grad_tensor(out_shape);
      AclDescHelper::SetFormat(y_grad_tensor.desc, ACL_FORMAT_NC1HWC0);
      AclDescHelper::SetFormat(x_grad_tensor.desc, ACL_FORMAT_NC1HWC0);
      NpuTensor<float> scale_tensor({32});
      NpuTensor<float> reserve_space_1({32});
      NpuTensor<float> reserve_space_2({32});
      NpuTensor<float> scale_grad_tensor({32});
      NpuTensor<float> offset_grad_tensor({32});
      NpuTensor<float> reserve_space_3({32});
      NpuTensor<float> reserve_space_4({32});
      {
        NpuRunner runner("BatchNormGrad");
        runner.AddInput(y_grad_tensor)
        .AddInput(out_tensor)
        .AddInput(scale_tensor)
        .AddInput(reserve_space_1)
        .AddInput(reserve_space_2)
        .AddOutput(x_grad_tensor)
        .AddOutput(scale_grad_tensor)
        .AddOutput(offset_grad_tensor)
        .AddOutput(reserve_space_3)
        .AddOutput(reserve_space_4)
        .SetAttr("data_format", "NCHW")
        .SetAttr("epsilon", 1e-7f)
        .SetAttr("is_training", true)
        .Run();
      }
      {
        NpuRunner runner("BatchNormGrad");
        runner.AddInput(y_grad_tensor)
        .AddInput(out_tensor)
        .AddInput(scale_tensor)
        .AddInput(reserve_space_1)
        .AddInput(reserve_space_2)
        .AddOutput(x_grad_tensor)
        .AddOutput(scale_grad_tensor)
        .AddOutput(offset_grad_tensor)
        .AddOutput(reserve_space_3)
        .AddOutput(reserve_space_4)
        .SetAttr("data_format", "NCHW")
        .SetAttr("epsilon", 1e-7f)
        .SetAttr("is_training", true)
        .Run();
      }
#endif
#if 1
      {
        NpuRunner runner("BatchNormGradExt2");
        runner.AddInput(y_grad_tensor)
        .AddInput(out_tensor)
        .AddInput(scale_tensor)
        .AddInput(reserve_space_1)
        .AddInput(reserve_space_2)
        .AddOutput(x_grad_tensor)
        .AddOutput(scale_grad_tensor)
        .AddOutput(offset_grad_tensor)
        .AddOutput(reserve_space_3)
        .AddOutput(reserve_space_4)
        .SetAttr("data_format", "NCHW")
        .SetAttr("epsilon", 1e-7f)
        .SetAttr("is_training", true)
        .Run();
      }
      {
        NpuRunner runner("BatchNormGradExt2");
        runner.AddInput(y_grad_tensor)
        .AddInput(out_tensor)
        .AddInput(scale_tensor)
        .AddInput(reserve_space_1)
        .AddInput(reserve_space_2)
        .AddOutput(x_grad_tensor)
        .AddOutput(scale_grad_tensor)
        .AddOutput(offset_grad_tensor)
        .AddOutput(reserve_space_3)
        .AddOutput(reserve_space_4)
        .SetAttr("data_format", "NCHW")
        .SetAttr("epsilon", 1e-7f)
        .SetAttr("is_training", true)
        .Run();
      }
#endif
    }  
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
