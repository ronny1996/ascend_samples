#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<float> x_tensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    NpuTensor<int64_t> perm_tensor({3}, {0, 2, 1});
    NpuTensor<float> out_tensor({2, 2, 2});
    {
      NpuRunner runner("Transpose");
      runner.AddInput(x_tensor).AddInput(perm_tensor)
          .AddOutput(out_tensor)
          .Run(); //.SetAttr("axes", static_cast<int64_t>(0));
    }
    out_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
