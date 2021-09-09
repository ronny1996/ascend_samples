#include "initializer_list"
#include "npu/npu_device.h"
#include "npu/npu_internal_helper.h"
#include "npu/npu_runner.h"

#define DTYPE float

int main(int argc, char const* argv[]) {
  /* code */
  NpuDeviceHelper::InitAllDevices();
  NpuDeviceHelper::SetCurrentDevice(0);
  {
    // {
    //   NpuTensor<DTYPE> x_tensor({2, 4}, std::initializer_list<DTYPE>({1, 2,
    //   3, -4, 5, -6, 7, -8})); NpuTensor<DTYPE> out_tensor({2, 4});
    //   {
    //     NpuGuard guard(0);
    //     auto builder =
    //     NpuRunner::Builder("Axpb").AddInput(x_tensor).AddOutput(out_tensor);
    //     NpuRunner runner(builder);
    //     runner.SetAttr("alpha", 2.0f).SetAttr("beta", 5.0f);
    //     runner.Run(guard.GetStream());
    //   }
    //   out_tensor.print();
    // }

    // {
    //   NpuTensor<DTYPE> x_tensor({2, 4}, std::initializer_list<DTYPE>({1, 2,
    //   3, -4, 5, -6, 7, -8})); NpuTensor<DTYPE> y_tensor({4},
    //   std::initializer_list<DTYPE>({1, 1, 1, 1})); NpuTensor<DTYPE>
    //   dy_tensor({4},  std::initializer_list<DTYPE>({1, 1, 1, 1}));
    //   NpuTensor<DTYPE> dx_tensor({2, 4}, std::initializer_list<DTYPE>({1, 2,
    //   3, -4, 5, -6, 7, -8}));
    //   {
    //     NpuGuard guard(0);
    //     auto builder =
    //     NpuRunner::Builder("LpNormGrad").AddInput(x_tensor).AddInput(y_tensor).AddInput(dy_tensor).AddOutput(dx_tensor);
    //     NpuRunner runner(builder);
    //     runner.SetAttr("p", 4).SetAttr("axes", {0}).SetAttr("epsilon",
    //     1e-12f).SetAttr("keepdims", false); runner.Run(guard.GetStream());
    //   }
    //   x_tensor.print();
    //   y_tensor.print();
    //   dy_tensor.print();
    //   dx_tensor.print();
    // }

    {
      NpuTensor<DTYPE> x_tensor(
          {2, 2, 2}, std::initializer_list<DTYPE>({1, 2, 3, -4, 5, -6, 7, -8}));
      NpuTensor<int32_t> indices({1, 2},
                                 std::initializer_list<int32_t>({0, 0}));
      NpuTensor<DTYPE> updates({1, 2},
                               std::initializer_list<DTYPE>({1, 2}));
      NpuTensor<DTYPE> output({2, 2, 2});
      {
        NpuGuard guard(0);
        auto builder = NpuRunner::Builder("ScatterNdAdd")
                           .AddInput(x_tensor)
                           .AddInput(indices)
                           .AddInput(updates)
                           .AddOutput(output);
        NpuRunner runner(builder);
        runner.SetAttr("use_locking", false);
        runner.Run(guard.GetStream());
      }
      x_tensor.print();
      output.print();
    }
  }
  NpuDeviceHelper::ReleaseAllDevices();
  return 0;
}
