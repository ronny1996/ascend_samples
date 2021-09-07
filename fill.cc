#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<float> out_tensor({2, 4});
    {
      NpuRunner runner("FillV2D");
      runner.AddOutput(out_tensor)
      .SetAttr("value", 2.0f)
      .SetAttr("dims", {2, 4})
          .Run();
    }
    out_tensor.print();
  }
  {
    NpuTensor<double> out_tensor({2, 4});
    NpuTensor<int32_t> dims({2}, {2, 4});
    NpuTensor<double> value({1}, {2.0f});
    {
      NpuRunner runner("Fill");
      runner.AddInput(dims)
      .AddInput(value)
      .AddOutput(out_tensor)
          .Run();
    }
    out_tensor.print();
  }
  {
    NpuTensor<float> out_tensor({64});
    NpuTensor<float> value({1}, {2.0f});
    {
      NpuRunner runner("FillD");
      runner.AddInput(value)
      .AddOutput(out_tensor)
      .SetAttr("dims", static_cast<int>(64))
          .Run();
    }
    out_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
