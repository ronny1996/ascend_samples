#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  { // ExpandD + TileWithAxis = Broadcast
    NpuTensor<float> x_tensor({2, 3}, {1, 2, 3, 4, 5, 6});
    NpuTensor<float> out_tensor({2, 2, 3, 1});
    {
      NpuRunner runner("ExpandD");
      runner.AddInput(x_tensor)
          .AddOutput(
          out_tensor)
          .SetAttr("shape", std::vector<int64_t>({2, 2, 3}))
          .Run();
    }
    out_tensor.print();
    NpuTensor<float> tile_out_tensor({2, 2, 3, 1});
    {
      NpuRunner runner("TileWithAxis");
      runner.AddInput(out_tensor)
      .AddOutput(tile_out_tensor)
      .SetAttr("axis", static_cast<int64_t>(3))
      .SetAttr("tiles",static_cast<int64_t>(1))
      .Run();
    }
    tile_out_tensor.print();
  }  // ~NpuTensor
  NpuHelper::ReleaseAllDevices();
  return 0;
}
