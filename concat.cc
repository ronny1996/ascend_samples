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
      NpuTensorHelper::SetName(x_tensor, "x0");
      NpuTensorHelper::SetName(y_tensor, "x1");


      NpuTensor<float> out_tensor({2, 8});
      // NpuTensor<cosnt int32_t> concat_dim({1}, {0});
      {
        NpuRunner runner("ConcatD");
        runner.AddInput(x_tensor)
            .AddInput(y_tensor)
            // .AddInput(concat_dim)
            .AddOutput(out_tensor)
            .SetAttr("concat_dim", 1)
            .SetAttr("N", 2)
            .Run();
      }
      out_tensor.print();
    }
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
