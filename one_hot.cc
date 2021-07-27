#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<int32_t> x_tensor({4}, {1, 1, 3, 4});

    NpuTensor<const int32_t> depth({1}, {5});
    NpuTensor<const float> on_value({1}, {1});
    NpuTensor<const float> off_value({1}, {0});
    NpuTensor<float> out_tensor({4, 6});
    {
      NpuRunner runner("OneHot");
      runner.AddInput(x_tensor).AddInput(depth).AddInput(on_value).AddInput(off_value)
          .AddOutput(out_tensor)
          .SetAttr("axis", -1)
          .Run();
    }
    out_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
