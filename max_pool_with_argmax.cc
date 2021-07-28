#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices({0, 1, 2, 3, 4, 5, 7});
  NpuHelper::SetDevice(2);
  #if 0
  {
    int N = 1, C = 1, H = 8, W = 8;
    std::vector<int32_t> ksize({2, 2});
    std::vector<int32_t> strides({2, 2});
    std::vector<int32_t> pads({1, 1});
    auto Ho = (H + pads[0] * 2 - ksize[0]) / strides[0] + 1;
    auto Wo = (W + pads[1] * 2 - ksize[1]) / strides[1] + 1;
    std::vector<int64_t> x_shape({N, C, H, W});
    std::vector<int64_t> out_shape({N, C, Ho, Wo}); // (n + 2p - k) / s + 1
    NpuTensor<float> x_tensor(x_shape);
    NpuTensor<float> out_tensor(out_shape);
    NpuTensor<uint16_t> mask_tensor({N, C, Ho, Wo});
    {
      NpuRunner runner("MaxPoolWithArgmaxV1", 2);
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
          .AddOutput(mask_tensor)
          .SetAttr("ksize", {1, ksize[0], ksize[1], 1})
          .SetAttr("strides", {1, strides[0], strides[1], 1})
          .SetAttr("pads", {1, pads[0], pads[1], 1})
          // .SetAttr("dtype", static_cast<int32_t>(AclDataType<int32_t>::type))
          .SetAttr("dilation", {1, 1, 1, 1})
          .SetAttr("ceil_mode", false)
          .Run();
    }
    out_tensor.print();
    mask_tensor.print();
  }
  #endif
  {
    int N = 1, C = 1, H = 8, W = 8;
    std::vector<int32_t> ksize({2, 2});
    std::vector<int32_t> strides({2, 2});
    std::string paddings = "VALID";
    auto Ho = (H - ksize[0]) / strides[0] + 1;
    auto Wo = (W - ksize[1]) / strides[1] + 1;
    std::vector<int64_t> x_shape({N, C, H, W});
    std::vector<int64_t> out_shape({N, C, Ho, Wo});
    NpuTensor<float> x_tensor(x_shape);
    NpuTensor<float> out_tensor(out_shape);
    NpuTensor<uint16_t> mask_tensor({N, C, Ho, Wo});
    {
      NpuRunner runner("MaxPoolWithArgmax", 2);
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
          .AddOutput(mask_tensor)
          .SetAttr("ksize", {1, ksize[0], ksize[1], 1})
          .SetAttr("strides", {1, strides[0], strides[1], 1})
          .SetAttr("padding", "VALID")
          // .SetAttr("Targmax", static_cast<int32_t>(AclDataType<int32_t>::type))
          .SetAttr("dilation", {1, 1, 1, 1})
          .SetAttr("ceil_mode", false)
          .Run();
    }
    out_tensor.print();
    mask_tensor.print();
  }
  
  NpuHelper::ReleaseAllDevices({0, 1, 2, 3, 4, 5, 7});
  return 0;
}
