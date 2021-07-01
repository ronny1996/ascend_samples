#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<int> shape_tensor({2}, {2, 3});
    NpuTensor<float> out_tensor({2, 3});
    {
      NpuRunner runner("StandardNormal");
      runner.AddInput(shape_tensor)
          .AddOutput(out_tensor)
          .SetAttr("dtype", static_cast<int64_t>(ACL_FLOAT))
          .Run();
    }
    out_tensor.print();
  }
  {
    NpuTensor<int> shape_tensor({2}, {2, 3});
    NpuTensor<float> out_tensor({2, 3});
    {
      NpuRunner runner("RandomUniform");
      runner.AddInput(shape_tensor)
          .AddOutput(out_tensor)
          .SetAttr("dtype", static_cast<int64_t>(ACL_FLOAT))
          .Run();
    }
    out_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
