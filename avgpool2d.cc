#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<float> x_tensor({1, 1, 2, 2}, {1, 5, 2, 2});
    NpuTensor<float> out_tensor({1, 1, 1, 1});
    {
      NpuRunner runner("AvgPool");
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
          .SetAttr("ksize", std::vector<int64_t>({1, 1, 2, 2})) // nchw
          .SetAttr("strides", std::vector<int64_t>({1, 1, 1, 1}))
          .SetAttr("padding", "VALID")
          .SetAttr("data_format", "NCHW")  // data-format of ksize and strides.
          .Run();
    }
    out_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
