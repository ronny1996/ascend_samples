#include "npu/npu_internal_helper.h"
#include "npu/npu_device.h"
#include "npu/npu_runner.h"

int64_t CeilDiv(int64_t value, int64_t factor) {
  int64_t value_num = 0;
  if (factor == 0) {
    return value_num;
  }
  if (value % factor == 0) {
    value_num = value / factor;
  } else {
    value_num = value / factor + 1;
  }
  return value_num;
}

int main(int argc, char const* argv[]) {
  /* code */
  NpuDeviceHelper::InitAllDevices();
  NpuDeviceHelper::SetCurrentDevice(0);
  {
    {
      int N = 1, C = 1, H = 4, W = 4;
      std::vector<int32_t> ksize({2, 2});
      std::vector<int32_t> strides({2, 2});
      std::vector<int32_t> pads({0, 0});
      auto Ho = (H + pads[0] * 2 - ksize[0]) / strides[0] + 1;
      auto Wo = (W + pads[1] * 2 - ksize[1]) / strides[1] + 1;
      std::vector<int64_t> x_shape({N, C, H, W});
      std::vector<int64_t> out_shape({N, C, Ho, Wo}); // (n + 2p - k) / s + 1
      std::vector<float> x_data;
      for (auto i = 0; i < N * C * H * W; i++) {
        x_data.push_back(static_cast<float>((rand() % 20 - 10) * 1.0 / 10));
      }
      NpuTensor<float> x_tensor(x_shape, x_data);
      NpuTensor<float> out_tensor(out_shape);
      const int64_t BLOCKSIZE = 16;
      int64_t maskH = ksize[0] * ksize[1];
      int64_t maskW = CeilDiv(Ho * Wo, BLOCKSIZE);
      NpuTensor<uint16_t> mask_tensor({N, C, maskH, maskW});

      {
        NpuGuard guard(0);
        auto builder = NpuRunner::Builder("MaxPoolWithArgmaxV2").AddInput(x_tensor).AddOutput(out_tensor).AddOutput(mask_tensor);
        NpuRunner runner(builder);
        runner.SetAttr("ksize", {1, ksize[0], ksize[1], 1})
          .SetAttr("strides", {1, strides[0], strides[1], 1})
          .SetAttr("pads", {1, pads[0], pads[1], 1})
          .SetAttr("dilation", {1, 1, 1, 1})
          .SetAttr("ceil_mode", false);
        runner.Run(guard.GetStream());
      }
      out_tensor.print();
      mask_tensor.print();
    }
  }
  NpuDeviceHelper::ReleaseAllDevices();
  return 0;
}
