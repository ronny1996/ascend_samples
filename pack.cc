#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    // NpuHelper::Profiler prof("/work/npu_prof/");
    {
      NpuTensor<float> x_tensor({2, 4}, {1, 2, 3, -4, 5, -6, 7, -8});
      NpuTensor<float> y_tensor({2, 4}, {1, 2, 3, -4, 5, -6, 7, -8});
      NpuTensor<float> out_tensor({2, 2, 4});
      {
        NpuRunner runner("Pack");
        runner.AddInput(x_tensor)
            .AddInput(y_tensor)
            .AddOutput(out_tensor)
            .SetAttr("axis", 0)
            .SetAttr("N", 2)
            .Run();
      }
      out_tensor.print();
    }
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
