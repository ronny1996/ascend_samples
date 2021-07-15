#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
#if 0
  {
    NpuTensor<float> x_tensor({2, 4}, {1, 2, 3, -4, 5, -6, 7, -8});

    NpuTensor<float> out_tensor({2, 4});

    {
      NpuRunner runner("Dropout");
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
        //   .SetAttr("dropout_ratio", 0.6f)
        //   .SetAttr("scale_train", true)
        //   .SetAttr("alpha", 1.0f)
        //   .SetAttr("beta", 0.0f)
          .Run();
    }
    out_tensor.print();
  }
#endif
#if 0
  {
    NpuTensor<float> x_tensor({2, 4}, {1, 2, 3, -4, 5, -6, 7, -8});
    NpuTensor<float> x_seed_tensor({2, 4}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8});

    NpuTensor<float> out_tensor({2, 4});
    NpuTensor<float> mask_tensor({2, 4});
    NpuTensor<float> seed_tensor({2, 4});

    {
      NpuRunner runner("DropoutV2");
      runner.AddInput(x_tensor)
          .AddInput(x_seed_tensor)
          .AddOutput(out_tensor)
          .AddOutput(mask_tensor)
          .AddOutput(seed_tensor)
          .SetAttr("p", 0.6f)
          .Run();
    }
    out_tensor.print();
    mask_tensor.print();
    seed_tensor.print();
  }
#endif
  {
    NpuTensor<int32_t> x_shape_tensor({1}, {2, 4});
    NpuTensor<float> prob_tensor({2, 4}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8});
    NpuTensor<uint8_t> out_tensor({2 * 4 * 16});
    {
      NpuRunner runner("DropOutGenMask");
      runner.AddInput(x_shape_tensor)
          .AddInput(prob_tensor)
          .AddOutput(out_tensor)
        //   .SetAttr("seed", 1)
        //   .SetAttr("seed2", 2)
          .Run();
    }
    out_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
