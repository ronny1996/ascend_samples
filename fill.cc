#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  {
    NpuTensor<double> out_tensor({2, 4});
    {
      NpuRunner runner("FillV2D");
      runner.AddOutput(out_tensor)
      .SetAttr("value", 2.0f)
      .SetAttr("dims", {2, 4})
          .Run();
    }
    out_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
