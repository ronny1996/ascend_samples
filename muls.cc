#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<float> x_tensor({2, 4}, {1, 2, 3, -4, 5, -6, 7, -8});
    NpuTensor<float> out_tensor({2, 4});
    {
      NpuRunner runner("Muls");
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
          .SetAttr("value", 2.0f)
          .Run();
    }
    out_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
