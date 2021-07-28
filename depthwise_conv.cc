#include "npu_runner.h"

#include <algorithm>

int main(int argc, char const* argv[]) {
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    int groups = 3;
    #if 0
    std::vector<int64_t> x_shape({2, 3, 5, 5});
    std::vector<int64_t> out_shape({2, 3, 3, 3});
    int Cin = x_shape[1];
    int Cout = out_shape[1];
    std::vector<int64_t> filter_shape({Cout, Cin / groups, 3, 3});
    #else
    std::vector<int64_t> x_shape({2, 3, 5, 5});
    std::vector<int64_t> out_shape({2, 3, 4, 4});
    int Cin = x_shape[1];
    int Cout = out_shape[1];
    std::vector<int64_t> filter_shape({1, 3, 2, 2});
    #endif
    size_t x_numel = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<int64_t>());
    size_t filter_numel = std::accumulate(filter_shape.begin(), filter_shape.end(), 1, std::multiplies<int64_t>());
    size_t out_numel = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int64_t>());

    std::vector<float> filter_data;
    for (auto i = 0; i < filter_numel; i++) {
      filter_data.push_back(i);
    }

    NpuTensor<float> x_tensor(x_shape, std::vector<float>(x_numel, 1.0f));
    NpuTensor<float> filter_tensor(filter_shape, filter_data); //ACL_FORMAT_FRACTAL_Z);
    NpuTensor<float> bias_tensor({1}, {0});
    NpuTensor<float> out_tensor(out_shape);
    {
      NpuRunner runner("DepthwiseConv2D");
      runner.AddInput(x_tensor)
          .AddInput(filter_tensor)
          .AddOutput(out_tensor)
          .SetAttr("strides", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input, n and c must be set to 1.
          .SetAttr("dilations", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input
          .SetAttr("pads", std::vector<int64_t>({0, 0, 0, 0})) // t, b, l, r
          // .SetAttr("groups", static_cast<int64_t>(groups))
          .SetAttr("data_format", "NCHW")
          .Run();
    }
    out_tensor.print();
#if 1
    NpuTensor<float> out_grad_tensor(out_shape, std::vector<float>(out_numel, 1.0f));
    NpuTensor<float> filter_grad_tensor(filter_shape);
    #if 1
    NpuTensor<const int32_t> filter_shape_tensor({4}, NpuHelper::ConvertVectorType<int32_t>(filter_shape), ACL_FORMAT_NCHW, ACL_MEMTYPE_HOST); 
    {
      NpuRunner runner("DepthwiseConv2DBackpropFilter");
      runner.AddInput(x_tensor)
          .AddInput(filter_shape_tensor)
          .AddInput(out_grad_tensor)
          .AddOutput(filter_grad_tensor)
          .SetAttr("strides", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input, n and c must be set to 1.
          .SetAttr("pads", std::vector<int64_t>({0, 0, 0, 0})) // t, b, l, r
          .SetAttr("dilations", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input
          // .SetAttr("groups", static_cast<int64_t>(groups))
          .SetAttr("data_format", "NCHW")
          .Run();
    }
    #else
    {
      NpuRunner runner("Conv2DBackpropFilterD");
      runner.AddInput(x_tensor)
          .AddInput(out_grad_tensor)
          .AddOutput(filter_grad_tensor)
          .SetAttr("filter_size", std::vector<int64_t>(filter_shape))
          .SetAttr("strides", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input, n and c must be set to 1.
          .SetAttr("pads", std::vector<int64_t>({0, 0, 0, 0})) // t, b, l, r
          .SetAttr("dilations", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input
          .SetAttr("groups", static_cast<int64_t>(groups))
          .SetAttr("data_format", "NCHW")
          .Run();
    }
    #endif
    filter_grad_tensor.print();
#endif
#if 1
    NpuTensor<float> x_grad_tensor(x_shape);
    NpuTensor<const int32_t> input_shape_tensor({4}, NpuHelper::ConvertVectorType<int32_t>(x_shape), ACL_FORMAT_NCHW, ACL_MEMTYPE_HOST); 
    #if 1
    {
      NpuRunner runner("DepthwiseConv2DBackpropInput");
      runner.AddInput(input_shape_tensor)
          .AddInput(filter_tensor)
          .AddInput(out_grad_tensor)
          .AddOutput(x_grad_tensor)
          .SetAttr("strides", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input, n and c must be set to 1.
          .SetAttr("pads", std::vector<int64_t>({0, 0, 0, 0})) // t, b, l, r
          .SetAttr("dilations", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input
          // .SetAttr("groups", static_cast<int64_t>(3))
          .SetAttr("data_format", "NCHW")
          .Run();
    }
    #else
    {
      NpuRunner runner("Conv2DBackpropInputD");
      runner.AddInput(filter_tensor)// .AddInput(filter_shape_tensor)
          .AddInput(out_grad_tensor)
          .AddOutput(x_grad_tensor)
          .SetAttr("input_size", std::vector<int64_t>({1, 1, 2, 2}))
          .SetAttr("strides", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input, n and c must be set to 1.
          .SetAttr("pads", std::vector<int64_t>({0, 0, 0, 0})) // t, b, l, r
          .SetAttr("dilations", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input
          .SetAttr("groups", static_cast<int64_t>(3))
          .SetAttr("data_format", "NCHW")
          .Run();
    }
    #endif
    x_grad_tensor.print();
#endif
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}

