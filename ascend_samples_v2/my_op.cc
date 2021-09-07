#include "npu/npu_internal_helper.h"
#include "npu/npu_device.h"
#include "npu/npu_runner.h"
#include "initializer_list"

#define DTYPE float

int main(int argc, char const* argv[]) {
  /* code */
  NpuDeviceHelper::InitAllDevices();
  NpuDeviceHelper::SetCurrentDevice(0);
  {
    // {
    //   NpuTensor<DTYPE> x_tensor({2, 4}, std::initializer_list<DTYPE>({1, 2, 3, -4, 5, -6, 7, -8}));
    //   NpuTensor<DTYPE> out_tensor({2, 4});
    //   {
    //     NpuGuard guard(0);
    //     auto builder = NpuRunner::Builder("Axpb").AddInput(x_tensor).AddOutput(out_tensor);
    //     NpuRunner runner(builder);
    //     runner.SetAttr("alpha", 2.0f).SetAttr("beta", 5.0f);
    //     runner.Run(guard.GetStream());
    //   }
    //   out_tensor.print();
    // }
    {
      NpuTensor<DTYPE> x_tensor({2, 4}, std::initializer_list<DTYPE>({1, 2, 3, -4, 5, -6, 7, -8}));
      NpuTensor<DTYPE> y_tensor({4},  std::initializer_list<DTYPE>({1, 1, 1, 1}));
      NpuTensor<DTYPE> dy_tensor({4},  std::initializer_list<DTYPE>({1, 1, 1, 1}));
      NpuTensor<DTYPE> dx_tensor({2, 4}, std::initializer_list<DTYPE>({1, 2, 3, -4, 5, -6, 7, -8}));
      {
        NpuGuard guard(0);
        auto builder = NpuRunner::Builder("LpNormGrad").AddInput(x_tensor).AddInput(y_tensor).AddInput(dy_tensor).AddOutput(dx_tensor);
        NpuRunner runner(builder);
        runner.SetAttr("p", 4).SetAttr("axes", {0}).SetAttr("epsilon", 1e-12f).SetAttr("keepdims", false);
        runner.Run(guard.GetStream());
      }
      x_tensor.print();
      y_tensor.print();
      dy_tensor.print();
      dx_tensor.print();
    }
  }
  NpuDeviceHelper::ReleaseAllDevices();
  return 0;
}
