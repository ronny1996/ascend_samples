#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  {
    NpuTensor<float> x_tensor({2, 4}, {1, 2, 3, -4, 5, -6, 7, -8});
    NpuTensor<float> out_tensor({2, 4});
    {
      NpuRunner runner("Abs");
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
          .Run();
    }
    out_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
