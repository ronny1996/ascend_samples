#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<float> x_tensor({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    NpuTensor<float> filter_tensor({1, 1, 2, 2}, {1, 2, 3, 4}); //ACL_FORMAT_FRACTAL_Z);
    NpuTensor<float> bias_tensor({1}, {0});

    NpuTensor<float> out_tensor({1, 1, 2, 2});
    {
      NpuRunner runner("Conv2D");
      runner.AddInput(x_tensor)
          .AddInput(filter_tensor)
          .AddOutput(out_tensor)
          .SetAttr("strides", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input, n and c must be set to 1.
          .SetAttr("pads", std::vector<int64_t>({0, 0, 0, 0})) // t, b, l, r
          .SetAttr("dilations", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input
          .SetAttr("groups", static_cast<int64_t>(1))
          .Run();
    }
    out_tensor.print();
#if 1
    NpuTensor<float> out_grad_tensor({1, 1, 2, 2}, {1, 1, 1, 1});
    NpuTensor<float> filter_grad_tensor({1, 1, 2, 2});
    NpuTensor<int32_t> filter_shape_tensor({4}, {1, 1, 2, 2});
    {
      NpuRunner runner("Conv2DBackpropFilterD");
      runner.AddInput(x_tensor)// .AddInput(filter_shape_tensor)
          .AddInput(out_grad_tensor)
          .AddOutput(filter_grad_tensor)
          .SetAttr("filter_size", std::vector<int64_t>({1, 1, 2, 2}))
          .SetAttr("strides", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input, n and c must be set to 1.
          .SetAttr("pads", std::vector<int64_t>({0, 0, 0, 0})) // t, b, l, r
          .SetAttr("dilations", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input
          .SetAttr("groups", 1L)
          .SetAttr("data_format", "NCHW")
          .Run();
    }
    filter_grad_tensor.print();

    NpuTensor<float> x_grad_tensor({1, 1, 3, 3});
    {
      NpuRunner runner("Conv2DBackpropInputD");
      runner.AddInput(filter_tensor)// .AddInput(filter_shape_tensor)
          .AddInput(out_grad_tensor)
          .AddOutput(x_grad_tensor)
          .SetAttr("input_size", std::vector<int64_t>({1, 1, 3, 3}))
          .SetAttr("strides", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input, n and c must be set to 1.
          .SetAttr("pads", std::vector<int64_t>({0, 0, 0, 0})) // t, b, l, r
          .SetAttr("dilations", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input
          .SetAttr("groups", 1L)
          .SetAttr("data_format", "NCHW")
          .Run();
    }
    x_grad_tensor.print();
#endif
  }
#if 0
  {
    NpuTensor<float> x_tensor({1, 4, 4, 1}, std::vector<float>(4 * 4, 1.0f), ACL_FORMAT_NHWC);
    NpuTensor<float> filter_tensor({1, 1, 2, 2}, {1, 2, 3, 4}); //ACL_FORMAT_FRACTAL_Z);
    NpuTensor<float> out_tensor({1, 3, 3, 1}, ACL_FORMAT_NHWC);
    {
      NpuRunner runner("Conv2D");
      runner.AddInput(x_tensor)
          .AddInput(filter_tensor)
          .AddOutput(out_tensor)
          .SetAttr("strides", std::vector<int64_t>({1, 2, 2, 1})) // dataformat is same with input, n and c must be set to 1.
          .SetAttr("pads", std::vector<int64_t>({1, 1, 1, 1})) // t, b, l, r
          .SetAttr("dilations", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input
          .SetAttr("groups", static_cast<int64_t>(1))
          .Run();
    }
    out_tensor.print();
  }
#endif
  NpuHelper::ReleaseAllDevices();
  return 0;
}

/*
0 0 0 0 0 0
0 1 1 1 1 0
0 1 1 1 1 0
0 1 1 1 1 0
0 1 1 1 1 0
0 0 0 0 0 0

4 7  3
6 10 4
2 3  1
*/

/*
g++ conv2d.cc -fpermissive \
-I/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/fwkacllib/include/ \
-L/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/fwkacllib/lib64/ \
-lascendcl -lacl_op_compiler

fwkacllib和acllib区别？
*/
