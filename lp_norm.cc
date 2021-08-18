#include "npu_runner.h"
#include "climits"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<float> x_tensor({2, 4}, {1, 2, 0, -4, 5, -6, 7, -8});
    NpuTensor<float> out_tensor({2,});
    {
      NpuRunner runner("LpNorm");
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
          .SetAttr("p", static_cast<int32_t>(INT32_MAX))
          .SetAttr("axes", std::vector<int32_t>({1}))
          .SetAttr("keepdim", false)
          .SetAttr("epsilon", static_cast<float>(1e-12))
          .Run();
    }
    out_tensor.print();
  }
  
  NpuHelper::ReleaseAllDevices();
  return 0;
}
