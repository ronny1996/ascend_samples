#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    {
      NpuTensor<int32_t> x_tensor({4}, {1, 2, 3, 0});
      NpuTensor<int32_t> axis({1}, {0});
      NpuTensor<int32_t> num_lower({1}, {1});
      NpuTensor<int32_t> num_upper({1}, {4});
      NpuTensor<int32_t> y({4});
      NpuTensor<int32_t> index({4});
      NpuTensor<int32_t> count({4});

      {
        NpuRunner runner("UniqueWithCountsExt2");
        runner.AddInput(x_tensor).AddInput(axis)
            .AddOutput(y).AddOutput(index).AddOutput(count).SetAttr("out_idx", static_cast<int32_t>(AclDataType<int32_t>::type))
            .Run();
      }
      y.print();
    }
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
