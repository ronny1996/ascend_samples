#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    {
      NpuTensor<int32_t> x_tensor({8}, {1, 2, 0, 0, 3, 5, 6, 0});
      NpuTensor<int32_t> out_tensor({8});
      {
        NpuRunner runner("CumsumD");
        runner.AddInput(x_tensor)
            .AddOutput(out_tensor)
            .SetAttr("axis", 0)
            .Run();
      }
      out_tensor.print();
      NpuTensor<int32_t> unique_y({5});
      NpuTensor<int32_t> index({8});
      {
        NpuRunner runner("Unique");
        runner.AddInput(out_tensor).AddOutput(unique_y).AddOutput(index).Run();
      }
      unique_y.print();
      index.print();
    }
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
