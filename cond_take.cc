#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    // NpuTensor<float> x_tensor({2, 4}, {1, 2, 3, -4, 5, -6, 7, -8});
    // NpuTensor<float> zeros({2, 4});
    // NpuTensor<int32_t> mask_tensor({2,4});
    // {
    //   NpuRunner runner("Greater");
    //   runner.AddInput(x_tensor)
    //       .AddInput(zeros)
    //       .AddOutput(mask_tensor)
    //       .Run();
    // }
    // mask_tensor.print();

    {
      NpuTensor<float> x_tensor({3, 4}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
      NpuTensor<float> mask_tensor({3, 4}, {1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0}, ACL_FORMAT_NCHW, ACL_MEMTYPE_DEVICE);
      NpuTensor<float> out_tensor({3, 4});
      NpuTensor<int32_t> index_tensor({3, 4});
      NpuTensor<int32_t> valid_num({3, 4});
      {
        NpuRunner runner("CondTake");
        runner.AddInput(x_tensor).AddInput(mask_tensor)
            .AddOutput(out_tensor).AddOutput(index_tensor).AddOutput(valid_num)
            .Run();
      }
      out_tensor.print();
    }
  }
  
  NpuHelper::ReleaseAllDevices();
  return 0;
}
