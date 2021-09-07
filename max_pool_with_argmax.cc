#include "npu_runner.h"


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
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  #if 1
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

    auto shape4Dto5D = [](const std::vector<int64_t> &ori_shape) {
      const int64_t BLOCKSIZE = 16;
      return std::vector<int64_t>({ori_shape[0], (ori_shape[1] + BLOCKSIZE - 1) / BLOCKSIZE, ori_shape[2], ori_shape[3], BLOCKSIZE});
    };
    auto transdata_op = [](AclTensor& in_tensor, AclTensor& out_tensor, const std::string &src_format, const std::string &dst_format) {
      NpuRunner runner("TransData");
      runner.AddInput(in_tensor).AddOutput(out_tensor).SetAttr("src_format", src_format.c_str()).SetAttr("dst_format", dst_format.c_str()).Run();
    };
    auto cast_op = [](AclTensor& in_tensor, AclTensor& out_tensor, int dtype) {
      NpuRunner runner("Cast");
      runner.AddInput(in_tensor).AddOutput(out_tensor).SetAttr("dst_type", dtype).Run();
    };
    NpuTensor<float> x_tensor_transformed(shape4Dto5D(x_tensor.dims), ACL_FORMAT_NC1HWC0);
    NpuTensor<float> out_tensor_transformed(shape4Dto5D(out_tensor.dims), ACL_FORMAT_NC1HWC0);
    NpuTensor<uint16_t> mask_tensor_transformed(shape4Dto5D(mask_tensor.dims), ACL_FORMAT_NC1HWC0);
    NpuTensor<int32_t> mask_tensor_transformed2(shape4Dto5D(mask_tensor.dims), ACL_FORMAT_NC1HWC0);
    transdata_op(x_tensor, x_tensor_transformed, "NCHW", "NC1HWC0");
    {
      NpuRunner runner("MaxPoolWithArgmaxV1");
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor_transformed)
          .AddOutput(mask_tensor_transformed)
          .SetAttr("ksize", {1, ksize[0], ksize[1], 1})
          .SetAttr("strides", {1, strides[0], strides[1], 1})
          .SetAttr("pads", {1, pads[0], pads[1], 1})
          // .SetAttr("dtype", static_cast<int32_t>(AclDataType<int32_t>::type))
          .SetAttr("dilation", {1, 1, 1, 1})
          .SetAttr("ceil_mode", false)
          .Run();
    }
    x_tensor.print();
    out_tensor_transformed.print();
    mask_tensor_transformed.print();
    // transdata_op(out_tensor_transformed, out_tensor, "NC1HWC0", "NCHW");
    // cast_op(mask_tensor_transformed, mask_tensor_transformed2, AclDataType<int32_t>::type);

    out_tensor.print();
    mask_tensor.print();
  }
  #endif
  #if 0
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
    NpuTensor<int64_t> mask_tensor({N, C, Ho, Wo});
    {
      NpuRunner runner("MaxPoolWithArgmax");
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
  #endif
  NpuHelper::ReleaseAllDevices();
  return 0;
}
