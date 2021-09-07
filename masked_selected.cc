#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    {
      NpuTensor<float> x_tensor({3, 4}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
      NpuTensor<uint8_t> mask_tensor({3, 4}, {1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0}, ACL_FORMAT_NCHW, ACL_MEMTYPE_DEVICE, ACL_BOOL);
      NpuTensor<float> out_tensor({3, 4});
      {
        NpuRunner runner("MaskedSelectV2");
        runner.AddInput(x_tensor)
            .AddInput(mask_tensor)
            .AddOutput(out_tensor)
            .Run();
      }
      out_tensor.print();
    }


    // {
    //   NpuTensor<float> x_tensor({4}, {1,2,0,3});
    //   NpuTensor<float> out_tensor({3});
    //   NpuRunner runner("NonZero");
    //   runner.AddInput(x_tensor).AddOutput(out_tensor).Run();
    //   out_tensor.print();
    // }
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
