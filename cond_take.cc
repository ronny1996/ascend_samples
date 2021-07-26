#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<float> x_tensor({2, 4}, {1, 2, 3, -4, 5, -6, 7, -8});
    NpuTensor<float> zeros({2, 4});
    NpuTensor<int32_t> mask_tensor({2,4});
    {
      NpuRunner runner("Greater");
      runner.AddInput(x_tensor)
          .AddInput(zeros)
          .AddOutput(mask_tensor)
          .Run();
    }
    mask_tensor.print();
    // NpuTensor<float> out_tensor({2, 4});
    // {
    //   NpuRunner runner("CondTake");
    //   runner.AddInput(x_tensor)
    //       .AddOutput(out_tensor)
    //       .Run();
    // }
    // out_tensor.print();
  }
  
  NpuHelper::ReleaseAllDevices();
  return 0;
}
