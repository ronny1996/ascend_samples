#include "npu/npu_device.h"
#include "npu/npu_internal_helper.h"
#include "npu/npu_runner.h"

int main(int argc, char const* argv[]) {
  /* code */
  NpuDeviceHelper::InitAllDevices();
  NpuDeviceHelper::SetCurrentDevice(0);
  {
    {
      NpuTensor<float> x_tensor({2, 3, 3, 4}, std::vector<float>(2 * 3 * 3 * 4, 0.0f), ACL_FORMAT_NHWC);
      NpuTensor<float> y_tensor({4}, std::vector<float>({1, 2, 3, 4}), ACL_FORMAT_NHWC);
      NpuTensor<float> out_tensor({2, 3, 3, 4}, ACL_FORMAT_NHWC);
      NpuTensor<float> out_tensor2({2, 3, 3, 4}, ACL_FORMAT_NHWC);
      NpuTensor<float> out_tensor3({2, 3, 3, 4}, ACL_FORMAT_NHWC);

      {
        NpuGuard guard(0);
        NpuRunner runner(NpuRunner::Builder("Add")
                             .AddInput(x_tensor)
                             .AddInput(y_tensor)
                             .AddOutput(out_tensor.SetFormat("NCHW")));      // trans to 0,1,3,2
        runner.Run(guard.GetStream());
      }
      out_tensor.print();

      {
        NpuGuard guard(0);
        NpuRunner runner(NpuRunner::Builder("Add")
                             .AddInput(x_tensor)
                             .AddInput(y_tensor)
                             .AddOutput(out_tensor2.SetOriginFormat("NCHW"))); // trans to 0,3,1,2
        runner.Run(guard.GetStream());
      }
      out_tensor2.print();

      {
        NpuGuard guard(0);
        NpuRunner runner(NpuRunner::Builder("Add")
                             .AddInput(x_tensor)
                             .AddInput(y_tensor)
                             .AddOutput(out_tensor3.SetStorageFormat("NCHW")));
        runner.Run(guard.GetStream());
      }
      out_tensor3.print();
    }
  }
  NpuDeviceHelper::ReleaseAllDevices();
  return 0;
}

// [[[1, 2, 3, 4],
//   [1, 2, 3, 4],
//   [1, 2, 3, 4]],
//  [[1, 2, 3, 4],
//   [1, 2, 3, 4],
//   [1, 2, 3, 4]],
//  [[1, 2, 3, 4],
//   [1, 2, 3, 4],
//   [1, 2, 3, 4]]]
//
// [[[1, 2, 3, 4],
//   [1, 2, 3, 4],
//   [1, 2, 3, 4]],
//  [[1, 2, 3, 4],
//   [1, 2, 3, 4],
//   [1, 2, 3, 4]],
//  [[1, 2, 3, 4],
//   [1, 2, 3, 4],
//   [1, 2, 3, 4]]]