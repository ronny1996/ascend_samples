#include "npu_runner.h"
#include "climits"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<float> x_tensor({2, 4});
    NpuTensor<float> updates_tensor({2, 3}, {1, 0, 2, 4, 1, 1});
    NpuTensor<int32_t> indices_tensor({3}, {0, 0, 3});
    auto &out_tensor = x_tensor;
    {
      NpuRunner runner("InplaceIndexAdd");
      runner.AddInput(x_tensor)
          .AddInput(indices_tensor)
          .AddInput(updates_tensor)
          .AddOutput(out_tensor)
          .SetAttr("axis", 1)
          .Run();
    }
    out_tensor.print();
  }
  
  NpuHelper::ReleaseAllDevices();
  return 0;
}
