#include "npu/npu_internal_helper.h"
#include "npu/npu_device.h"
#include "npu/npu_runner.h"

int main(int argc, char const* argv[]) {
  /* code */
  NpuDeviceHelper::InitAllDevices();
  NpuDeviceHelper::SetCurrentDevice(0);
  {
    {
      NpuTensor<float> x_tensor({2, 4}, {1, 2, 3, -4, 5, -6, 7, -8});
      NpuTensor<float> out_tensor({2, 4});
      {
        NpuGuard guard(0);
        auto builder = NpuRunner::Builder("Abs").AddInput(x_tensor).AddOutput(out_tensor);
        NpuRunner runner(builder);
        runner.Run(guard.GetStream());
      }
      out_tensor.print();
    }
  }
  NpuDeviceHelper::ReleaseAllDevices();
  return 0;
}
