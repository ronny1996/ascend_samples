#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<float> x_tensor({2, 4}, {3.1415926, 2, 3.2, -4, 5, -6, 7, -8});
    NpuTensor<float> y_tensor({2, 4}, {3.1415926, 2.4, 3, -4, 5, -6.6, 7, -8});
    NpuTensor<uint8_t> out_tensor({2, 4});
    {
      NpuRunner runner("Equal");
      runner.AddInput(x_tensor)
          .AddInput(y_tensor)
          .AddOutput(out_tensor)
          .Run();
    }
    out_tensor.print();
  }
  
  NpuHelper::ReleaseAllDevices();
  return 0;
}
