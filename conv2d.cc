#include "npu_runner.h"

#include <algorithm>

int main(int argc, char const* argv[]) {
  /* code */
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
    std::vector<int64_t> filter_shape({Cout, Cin / groups, 2, 2});
    #endif
    size_t x_numel = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<int64_t>());
    size_t filter_numel = std::accumulate(filter_shape.begin(), filter_shape.end(), 1, std::multiplies<int64_t>());
    size_t out_numel = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int64_t>());

    NpuTensor<float> x_tensor(x_shape, std::vector<float>(x_numel, 1.0f));
    NpuTensor<float> filter_tensor(filter_shape, std::vector<float>(filter_numel, 1.0f)); //ACL_FORMAT_FRACTAL_Z);
    NpuTensor<float> bias_tensor({1}, {0});

    NpuTensor<float> out_tensor(out_shape);
    {
      NpuRunner runner("Conv2D");
      runner.AddInput(x_tensor)
          .AddInput(filter_tensor)
          .AddOutput(out_tensor)
          .SetAttr("strides", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input, n and c must be set to 1.
          .SetAttr("pads", std::vector<int64_t>({0, 0, 0, 0})) // t, b, l, r
          .SetAttr("dilations", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input
          .SetAttr("groups", static_cast<int64_t>(groups))
          .SetAttr("data_format", "NCHW")
          .Run();
    }
    out_tensor.print();
#if 1
    NpuTensor<float> out_grad_tensor(out_shape, std::vector<float>(out_numel, 1.0f));
    NpuTensor<float> filter_grad_tensor(filter_shape);
    std::vector<int32_t> tmp;
    for (auto t : filter_shape) {
      tmp.push_back(t);
    }    
    NpuTensor<const int32_t> filter_shape_tensor({4}, tmp, ACL_FORMAT_NCHW, ACL_MEMTYPE_HOST); 
    {
      #if 1
      NpuRunner runner("Conv2DBackpropFilter");
      runner.AddInput(x_tensor)
          .AddInput(filter_shape_tensor)
          .AddInput(out_grad_tensor)
          .AddOutput(filter_grad_tensor)
          .SetAttr("strides", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input, n and c must be set to 1.
          .SetAttr("pads", std::vector<int64_t>({0, 0, 0, 0})) // t, b, l, r
          .SetAttr("dilations", std::vector<int64_t>({1, 1, 1, 1})) // dataformat is same with input
          .SetAttr("groups", static_cast<int64_t>(groups))
          .SetAttr("data_format", "NCHW")
          .Run();
      #else
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
      #endif
    }
    filter_grad_tensor.print();
#endif
#if 0
    NpuTensor<float> x_grad_tensor({1, 1, 3, 3});
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
3,3,5,5
3,1,2,2
2,3,4,4
fwkacllib和acllib区别？
*/
