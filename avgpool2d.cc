#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);


  std::vector<int64_t> x_shape({2, 3, 4, 4});
  std::vector<int64_t> out_shape({2, 3, 2, 2}); // (n + 2p - k) / s + 1
  std::vector<float> x_data(2 * 3 * 4 * 4);
  for (auto i = 0; i < x_data.size(); i++) {
    x_data[i] = i;
  }
  {
    NpuTensor<float> x_tensor(x_shape, x_data);
    NpuTensor<float> out_tensor(out_shape);
    {
      NpuRunner runner("AvgPoolV2");
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
          .SetAttr("ksize", {1, 1, 2, 2}) // nchw
          .SetAttr("strides", {1, 1, 2, 2})
          .SetAttr("padding_mode", "CALCULATED")
          .SetAttr("pads", {0, 0, 0, 0})
          .SetAttr("data_format", "NCHW")  // data-format of ksize and strides.
          .SetAttr("global_pooling", false)
          .SetAttr("ceil_mode", false) // 0 - floor, 1 - ceil
          .SetAttr("exclusive", true)
          .Run();
    }
    out_tensor.print();
#if 1
    NpuTensor<float> x_grad_tensor(x_shape);
    NpuTensor<float> out_grad_tensor(out_shape,  std::vector<float>(2 * 3 * 2 * 2, 1.0f));
    NpuTensor<const int32_t> x_shape_tensor({4}, {2, 3, 4, 4}, ACL_FORMAT_NCHW, ACL_MEMTYPE_HOST); 
    {
      NpuRunner runner("AvgPoolV2Grad");
      runner
          .AddInput(x_shape_tensor)
          .AddInput(out_grad_tensor)
          .AddOutput(x_grad_tensor)
          // .SetAttr("orig_input_shape", x_shape)
          .SetAttr("ksize", std::vector<int64_t>({1, 1, 2, 2}))   // dims must be 4
          .SetAttr("strides", std::vector<int64_t>({1, 1, 2, 2})) // dims must be 4
          .SetAttr("padding_mode", "CALCULATED")
          .SetAttr("pads", {0, 0, 0, 0})
          .SetAttr("data_format", "NCHW")
          .SetAttr("global_pooling", false)
          .SetAttr("ceil_mode", false)
          .SetAttr("exclusive", true)
          .Run();
    }  
    x_grad_tensor.print();
#endif
  }
#if 0
  {
    std::swap(x_shape[1], x_shape[3]);
    std::swap(out_shape[1], out_shape[3]);
    NpuTensor<float> x_tensor(x_shape, x_data, ACL_FORMAT_NHWC);
    NpuTensor<float> out_tensor(out_shape, ACL_FORMAT_NHWC);
    {
      NpuRunner runner("AvgPoolV2");
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
          .SetAttr("ksize", {1, 2, 2, 1}) // nchw
          .SetAttr("strides", {1, 2, 2, 1})
          .SetAttr("padding_mode", "CALCULATED")
          .SetAttr("pads", {0, 0, 0, 0})
          .SetAttr("data_format", "NHWC")  // data-format of ksize and strides.
          .SetAttr("global_pooling", false)
          .SetAttr("ceil_mode", false)
          .SetAttr("exclusive", true)
          .Run();
    }
    out_tensor.print();
  }
#endif
  NpuHelper::ReleaseAllDevices();
  return 0;
}
