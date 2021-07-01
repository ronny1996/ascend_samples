#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<float> x_tensor({10}, {1, 3, 5, 4, 6, 2, 9, 7, 8, 10});
    NpuTensor<int32_t> k_tensor({1}, {5});
    NpuTensor<float> out_tensor({5});
    NpuTensor<int32_t> indices_tesnor({5});
    {
      NpuRunner runner("TopKD");
      runner.AddInput(x_tensor)
          .AddInput(k_tensor)
          .AddOutput(out_tensor)
          .AddOutput(indices_tesnor)
          .SetAttr("sorted", false) // must be false
          .SetAttr("largest", true)
          .SetAttr("dim", -1)
          .Run();
    }
    out_tensor.print();
    indices_tesnor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
