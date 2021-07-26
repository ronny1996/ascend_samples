#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<float> x_tensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    NpuTensor<const int32_t> perm_tensor({3}, {0, 2, 1});
    NpuTensor<float> out_tensor({2, 2, 2});
    {
      NpuRunner runner("Transpose");
      runner.AddInput(x_tensor).AddInput(perm_tensor)
          .AddOutput(x_tensor)
          .Run(); //.SetAttr("axes", static_cast<int64_t>(0));
    }
    x_tensor.print();
  }
  {
    NpuTensor<float> x_tensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    NpuTensor<float> out_tensor({2, 2, 2});
    {
      NpuRunner runner("TransposeD");
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
          .SetAttr("perm",  {0, 2, 1})
          .Run();
    }
    out_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
