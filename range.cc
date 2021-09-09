#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    {
      NpuTensor<int64_t> start({1}, {0});
      NpuTensor<int64_t> end({1}, {10});
      NpuTensor<int64_t> delta({1}, {1});
      NpuTensor<int64_t> out_tensor({2, 5});
      {
        NpuRunner runner("Range");
        runner.AddInput(start).AddInput(end).AddInput(delta)
            .AddOutput(out_tensor)
            .Run();
      }
      out_tensor.print();
    }
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
