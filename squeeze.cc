#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<float> x_tensor({2, 1, 4, 1}, {1, 2, 3, -4, 5, -6, 7, -8});
    NpuTensor<float> out_tensor({2, 4});
    {
      NpuRunner runner("Squeeze");
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
          .SetAttr("axis", {1, 3})
          .Run();
    }
    out_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
