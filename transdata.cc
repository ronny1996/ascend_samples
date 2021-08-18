#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<float> x_tensor({1, 3, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, -1, -2}, ACL_FORMAT_NCHW);
    NpuTensor<float> y_tensor({1, 2, 2, 3}, ACL_FORMAT_NHWC);
    {
      NpuRunner runner("TransData");
      runner.AddInput(x_tensor)
          .AddOutput(y_tensor)
          .SetAttr("src_format", "NCHW")
          .SetAttr("dst_format", "NHWC")
          .SetAttr("group", static_cast<int>(1))
          .Run();
    }
    x_tensor.print();
    y_tensor.print();
  }
  
  NpuHelper::ReleaseAllDevices();
  return 0;
}
