#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    {
      NpuTensor<float> x_tensor({4, 4}, std::vector<float>(4 * 4, 1));
      NpuTensor<float> y_tensor({4}, std::vector<float>({0, 1, 2, 3}));
      NpuTensor<float> out_tensor({4, 4});
      {
        NpuRunner runner("Add");
        runner.AddInput(x_tensor)
            .AddInput(y_tensor)
            .AddOutput(out_tensor)
            .Run();
      }
      out_tensor.print();
    }
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
