#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  // // avg pool
  // {
  //   std::vector<int64_t> x_tensor_shape({1, 4, 4, 1});
  //   NpuTensor<float> x_tensor(x_tensor_shape, ACL_FORMAT_NHWC);
  //   std::vector<int64_t> out_tensor_shape({1, 2, 2, 1});
  //   NpuTensor<float> out_tensor(out_tensor_shape, ACL_FORMAT_NHWC);
  //   std::vector<int64_t> ksize({2, 2});
  //   std::vector<int64_t> stride({2, 2});
  //   std::vector<int64_t> pad({0,0,0,0});
  //   {
  //     NpuRunner runner("Pooling");
  //     runner.AddInput(x_tensor)
  //         .AddOutput(out_tensor)
  //         .SetAttr("mode", static_cast<int64_t>(1)) // max:0 avg:1
  //         .SetAttr("global_pooling", false)
  //         .SetAttr("window", ksize)
  //         .SetAttr("stride", stride)
  //         .SetAttr("pad", pad)
  //         .SetAttr("dilation", std::vector<int64_t>({1, 1, 1, 1}))
  //         .SetAttr("ceil_mode", static_cast<int64_t>(0))
  //         .SetAttr("data_format", "NCHW") // required
  //         .Run();
  //   }
  //   out_tensor.print();

  //   NpuTensor<float> x_grad_tensor(x_tensor_shape, ACL_FORMAT_NHWC);
  //   NpuTensor<float> out_grad_tensor(out_tensor_shape, ACL_FORMAT_NHWC);
  //   // NpuTensor<const int32_t> inpu_shape_tensor({4}, {2, 2, 2, 2});
  //   NpuTensor<float> tmp1(out_tensor_shape, ACL_FORMAT_NHWC);
  //   NpuTensor<float> tmp2(out_tensor_shape, ACL_FORMAT_NHWC);
  //   {
  //     NpuRunner runner("AvgPoolV2GradD");
  //     runner
  //         // .AddInput(inpu_shape_tensor)
  //         .AddInput(out_grad_tensor)
  //         // .AddInput(tmp1)
  //         // .AddInput(tmp2)
  //         .AddOutput(x_grad_tensor)
  //         .SetAttr("orig_input_shape", x_tensor_shape)
  //         .SetAttr("ksize", std::vector<int64_t>({1,  ksize[0], ksize[1], 1}))   // dims must be 4
  //         .SetAttr("strides", std::vector<int64_t>({1, stride[0], stride[1], 1})) // dims must be 4
  //         .SetAttr("padding_mode", "CALCULATED")
  //         .SetAttr("pads", pad)
  //         .SetAttr("data_format", "NHWC")
  //         .SetAttr("global_pooling", false)
  //         .SetAttr("ceil_mode", false)
  //         .SetAttr("exclusive", true)
  //         .Run();
  //   }
  //   x_grad_tensor.print();
  // }
  // // max pool
  // {
  //   NpuTensor<float> x_tensor({1, 1, 2, 2}, {1, 5, 2, 2});
  //   NpuTensor<float> out_tensor({1, 1, 1, 1});
  //   {
  //     NpuRunner runner("Pooling");
  //     runner.AddInput(x_tensor)
  //         .AddOutput(out_tensor)
  //         .SetAttr("mode", static_cast<int64_t>(0)) // max:0 avg:1
  //         .SetAttr("global_pooling", false)
  //         .SetAttr("window", std::vector<int64_t>({2, 2}))
  //         .SetAttr("stride", std::vector<int64_t>({1, 1}))
  //         .SetAttr("pad", std::vector<int64_t>({0, 0, 0, 0}))
  //         .SetAttr("dilation", std::vector<int64_t>({1, 1, 1, 1}))
  //         .SetAttr("ceil_mode", static_cast<int64_t>(0))
  //         .SetAttr("data_format", "NCHW") // required
  //         .Run();
  //   }
  //   out_tensor.print();

  //   NpuTensor<float> x_grad_tensor({1, 1, 2, 2});
  //   NpuTensor<float> out_grad_tensor({1, 1, 1, 1}, {1});
  //   {
  //     NpuRunner runner("MaxPoolV3Grad");
  //     runner
  //         .AddInput(x_tensor)
  //         .AddInput(out_tensor)
  //         .AddInput(out_grad_tensor)
  //         .AddOutput(x_grad_tensor)
  //         .SetAttr("ksize", std::vector<int64_t>({1, 1, 2, 2}))   // dims must be 4
  //         .SetAttr("strides", std::vector<int64_t>({1, 1, 1, 1})) // dims must be 4
  //         .SetAttr("padding_mode", "CALCULATED")
  //         .SetAttr("pads", std::vector<int64_t>({0, 0, 0, 0})) // must less than ksize
  //         .SetAttr("data_format", "NCHW")
  //         .SetAttr("global_pooling", false)
  //         .SetAttr("ceil_mode", false)
  //         .Run();
  //   }
  //   x_grad_tensor.print();
  // }
  {
    std::vector<int64_t> x_tensor_shape({1, 4, 4, 1});
    NpuTensor<float> x_tensor(x_tensor_shape, ACL_FORMAT_NHWC);
    // NpuTensor<float> x_tensor(x_tensor_shape);
    std::vector<int64_t> out_tensor_shape({1, 2, 2, 1});
    NpuTensor<float> out_tensor(out_tensor_shape, ACL_FORMAT_NHWC);
    // NpuTensor<float> out_tensor(out_tensor_shape);
    std::vector<int64_t> ksize({2, 2});
    std::vector<int64_t> stride({2, 2});
    std::vector<int64_t> pad({0,0,0,0});
    {
      NpuRunner runner("Pooling");
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
          .SetAttr("mode", static_cast<int64_t>(1)) // max:0 avg:1
          .SetAttr("global_pooling", false)
          .SetAttr("window", ksize)
          .SetAttr("stride", stride)
          .SetAttr("pad", pad)
          .SetAttr("dilation", std::vector<int64_t>({1, 1, 1, 1}))
          .SetAttr("ceil_mode", static_cast<int64_t>(0)) // 1 - floor, 0 - ceil
          .SetAttr("data_format", "NCHW") // required
          .Run();
    }
    out_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
