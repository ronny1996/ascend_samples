#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<float> x_tensor({4}, {1, 1, 1, 1});
    NpuTensor<float> y_tensor({4}, {1, 0, 1, 0});
    NpuTensor<uint8_t> out_tensor({4});
    {
      NpuRunner runner("Greater");
      runner.AddInput(x_tensor)
          .AddInput(y_tensor)
          .AddOutput(out_tensor)
          .Run();
    }
    // out_tensor.print();
    out_tensor.sync();
    for (auto t : out_tensor.host_data) {
      std::cout << static_cast<int32_t>(t) << std::endl;
    }
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
