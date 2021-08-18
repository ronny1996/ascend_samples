#include "npu_runner.h"

#include <algorithm>

int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    // NpuHelper::Profiler profiler("/work/npu_prof/");
    {
      int groups = 1;
      std::vector<int64_t> x_shape({1, 2, 7, 7});
      std::vector<int64_t> out_shape({1, 3, 7, 7});
      int Cin = x_shape[1];
      int Cout = out_shape[1];
      std::vector<int64_t> filter_shape({Cout, Cin / groups, 1, 1});

      size_t x_numel = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<int64_t>());
      size_t filter_numel = std::accumulate(filter_shape.begin(), filter_shape.end(), 1, std::multiplies<int64_t>());
      size_t out_numel = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int64_t>());

      NpuTensor<float> x_tensor(x_shape, std::vector<float>(x_numel, 1.0f));
      NpuTensor<float> filter_tensor(filter_shape, std::vector<float>(filter_numel, 1.0f)); //ACL_FORMAT_FRACTAL_Z);
      NpuTensor<float> bias_tensor({1}, {0});
      NpuTensor<float> out_tensor(out_shape);


      // NpuTensor<npu::float16> x_tensor_fp16(x_shape);
      // NpuTensor<npu::float16> out_tensor_fp16(out_shape);
      // NpuTensor<npu::float16> filter_tensor_fp16(filter_shape);

      // NpuTensor<npu::float16> x_tensor_fp16_2({1,1,7,7,16}, ACL_FORMAT_NC1HWC0);
      // NpuTensor<npu::float16> out_tensor_fp16_2({1,1,7,7,16}, ACL_FORMAT_NC1HWC0);
      // NpuTensor<npu::float16> filter_tensor_fp16_2({1,1,16,16}, ACL_FORMAT_FRACTAL_Z);
      {
        auto trans_op = [](AclTensor& x, AclTensor&y, const std::string &src_fmt, const std::string &dst_fmt) {
          NpuRunner runner("TransData");
          runner.AddInput(x)
              .AddOutput(y)
              .SetAttr("src_format", src_fmt.c_str())
              .SetAttr("dst_format", dst_fmt.c_str())
              .SetAttr("group", static_cast<int>(1))
              .Run();
        };
        auto cast_op = [](AclTensor& x, AclTensor&y, const int32_t dtype) {
          NpuRunner runner("Cast");
          runner.AddInput(x)
              .AddOutput(y)
              .SetAttr("dst_type", dtype)
              .Run();
        };
        // {
        //   cast_op(filter_tensor, filter_tensor_fp16, static_cast<int32_t>(AclDataType<npu::float16>::type));
        //   trans_op(filter_tensor_fp16, filter_tensor_fp16_2, "NCHW", "FRACTAL_Z");
        //   cast_op(x_tensor, x_tensor_fp16, static_cast<int32_t>(AclDataType<npu::float16>::type));
        //   trans_op(x_tensor_fp16, x_tensor_fp16_2, "NCHW", "NC1HWC0");
        //   cast_op(out_tensor, out_tensor_fp16, static_cast<int32_t>(AclDataType<npu::float16>::type));
        //   trans_op(out_tensor_fp16, out_tensor_fp16_2, "NCHW", "NC1HWC0");
        // }
      }
#if 1
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
#else
      {
        NpuRunner runner("Conv2D");
        runner.AddInput(x_tensor_fp16_2)
            .AddInput(filter_tensor_fp16_2)
            .AddOutput(out_tensor_fp16_2)
            .SetAttr("strides", std::vector<int64_t>({1, 1, 1, 1}))
            .SetAttr("pads", std::vector<int64_t>({0, 0, 0, 0}))
            .SetAttr("dilations", std::vector<int64_t>({1, 1, 1, 1}))
            .SetAttr("groups", static_cast<int64_t>(groups))
            .SetAttr("data_format", "NC1HWC0")
            .Run();
      }
      {
        NpuRunner runner("Conv2D");
        runner.AddInput(x_tensor_fp16_2)
            .AddInput(filter_tensor_fp16_2)
            .AddOutput(out_tensor_fp16_2)
            .SetAttr("strides", std::vector<int64_t>({1, 1, 1, 1}))
            .SetAttr("pads", std::vector<int64_t>({0, 0, 0, 0}))
            .SetAttr("dilations", std::vector<int64_t>({1, 1, 1, 1}))
            .SetAttr("groups", static_cast<int64_t>(groups))
            .SetAttr("data_format", "NC1HWC0")
            .Run();
      }
#endif
  #if 0
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
    }  
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
