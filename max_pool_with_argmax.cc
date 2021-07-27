#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(2);
  {
    std::vector<int32_t> ksize({2, 2});
    std::vector<int32_t> strides({2, 2});
    std::vector<int32_t> pads({0, 0});
    std::vector<int64_t> x_shape({1, 8, 8, 1});
    auto Ho = (x_shape[2] + pads[0] * 2 - ksize[0]) / strides[0] + 1;
    auto Wo = (x_shape[3] + pads[1] * 2 - ksize[1]) / strides[1] + 1;
    std::vector<int64_t> out_shape({1, Ho, Wo, 1}); // (n + 2p - k) / s + 1

    NpuTensor<float> x_tensor(x_shape, ACL_FORMAT_NHWC);
    NpuTensor<float> out_tensor(out_shape, ACL_FORMAT_NHWC);
    NpuTensor<uint16_t> mask_tensor(out_shape, ACL_FORMAT_NHWC);
    {
      NpuRunner runner("MaxPoolWithArgmaxV1", 2);
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
          .AddOutput(mask_tensor)
          .SetAttr("ksize", {1, ksize[0], ksize[1], 1})
          .SetAttr("strides", {1, strides[0], strides[1], 1})
          .SetAttr("pads", {pads[0], pads[0], pads[1], pads[1]})
          .SetAttr("dtype", static_cast<int32_t>(AclDataType<float>::type))
          .SetAttr("dilation", {1, 1, 1, 1})
          .SetAttr("ceil_mode", false)
          .Run();
    }
    out_tensor.print();
    mask_tensor.print();
  }
  
  NpuHelper::ReleaseAllDevices();
  return 0;
}
