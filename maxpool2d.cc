#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);


  std::vector<int64_t> x_shape({1, 1, 5, 5});
  std::vector<int64_t> out_shape({1, 1, 1, 1}); // (n + 2p - k) / s + 1
  std::vector<float> x_data(1 * 1 * 5 * 5);
  for (auto i = 0; i < x_data.size(); i++) {
    x_data[i] = i;
  }
  {
    NpuTensor<float> x_tensor(x_shape, x_data);
    NpuTensor<float> out_tensor(out_shape);
    {
      NpuRunner runner("MaxPoolV3");
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
          .SetAttr("ksize", {1, 1, 3, 3}) // nchw
          .SetAttr("strides", {1, 1, 1, 1})
          .SetAttr("padding_mode", "CALCULATED")
          .SetAttr("pads", {0, 0, 0, 0})
          .SetAttr("data_format", "NCHW")  // data-format of ksize and strides.
          .SetAttr("global_pooling", true)
          .SetAttr("ceil_mode", false) // 0 - floor, 1 - ceil
          .SetAttr("exclusive", false)
          .Run();
    }
    out_tensor.print();

    NpuTensor<float> x_grad_tensor(x_shape);
    NpuTensor<float> out_grad_tensor(out_shape, std::vector<float>(1, 1.0f));
    {
      NpuRunner runner("MaxPoolV3Grad");
      runner
          .AddInput(x_tensor)
          .AddInput(out_tensor)
          .AddInput(out_grad_tensor)
          .AddOutput(x_grad_tensor)
          .SetAttr("ksize", std::vector<int64_t>({1, 1, 3, 3}))   // dims must be 4
          .SetAttr("strides", std::vector<int64_t>({1, 1, 1, 1})) // dims must be 4
          .SetAttr("padding_mode", "CALCULATED")
          .SetAttr("pads", std::vector<int64_t>({0, 0, 0, 0})) // must less than ksize
          .SetAttr("data_format", "NCHW")
          .SetAttr("global_pooling", true)
          .SetAttr("ceil_mode", false)
          .Run();
    }
    x_grad_tensor.print();
  }
  #if 0
  {
    std::swap(x_shape[1], x_shape[3]);
    std::swap(out_shape[1], out_shape[3]);
    NpuTensor<float> x_tensor(x_shape, x_data, ACL_FORMAT_NHWC);
    NpuTensor<float> out_tensor(out_shape, ACL_FORMAT_NHWC);
    {
      NpuRunner runner("MaxPoolV3");
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
