#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    // NpuHelper::Profiler prof("/work/npu_prof/");
    {
      NpuTensor<float> x_tensor({2, 3, 4, 5});
      NpuTensor<float> y_tensor({5});
      NpuTensor<float> out_tensor({2, 3, 4, 5});
      {
        NpuRunner runner("Mul");
        runner.AddInput(x_tensor)
            .AddInput(y_tensor)
            .AddOutput(out_tensor)
            .Run();
      }
      out_tensor.print();
    }
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
