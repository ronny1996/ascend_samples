#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);

  bool global_pool = false;
#if 1 //bad
  std::vector<int64_t> ksize({1, 1, 3, 3});
  std::vector<int64_t> strides({1, 1, 1, 2});
  std::vector<int64_t> pads({0, 0, 0, 0});
  std::vector<int64_t> x_shape({1, 1, 4, 5});
#else //good
  std::vector<int64_t> ksize({1, 1, 3, 3});
  std::vector<int64_t> strides({1, 1, 2, 2});
  std::vector<int64_t> pads({0, 0, 0, 0});
  std::vector<int64_t> x_shape({1, 1, 5, 5});
#endif
  // std::vector<int64_t> ksize({1, 1, 3, 3});
  // std::vector<int64_t> strides({1, 1, 1, 1});
  // std::vector<int64_t> pads({0, 0, 0, 0});
  // std::vector<int64_t> x_shape({2, 3, 5, 5});
  auto Ho = (x_shape[2] + pads[0] + pads[1] - ksize[2]) / strides[2] + 1;
  auto Wo = (x_shape[3] + pads[2] + pads[3] - ksize[3]) / strides[3] + 1;
  if (global_pool) {
    Ho = 1;
    Wo = 1;   
  }
  std::vector<int64_t> out_shape({x_shape[0], x_shape[1], Ho, Wo}); // (n + 2p - k) / s + 1
  size_t x_numel = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<int64_t>());
  size_t out_numel = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int64_t>());
  {
    NpuTensor<float> x_tensor(x_shape, std::vector<float>(x_numel, 1.0f));
    NpuTensor<float> out_tensor(out_shape);
    {
      NpuRunner runner("AvgPoolV2");
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
          .SetAttr("ksize", ksize) // nchw
          .SetAttr("strides", strides)
          .SetAttr("padding_mode", "CALCULATED")
          .SetAttr("pads", pads)
          .SetAttr("data_format", "NCHW")  // data-format of ksize and strides.
          .SetAttr("global_pooling", global_pool)
          .SetAttr("ceil_mode", false) // 0 - floor, 1 - ceil
          .SetAttr("exclusive", true)
          .Run();
    }
    out_tensor.print();
#if 1
    NpuTensor<float> x_grad_tensor(x_shape);
    NpuTensor<float> out_grad_tensor(out_shape,  std::vector<float>(out_numel, 1.0f));
    std::vector<int32_t> tmp;
    for (auto t : x_shape) tmp.push_back(t);
    NpuTensor<const int32_t> x_shape_tensor({4}, tmp, ACL_FORMAT_NCHW, ACL_MEMTYPE_HOST); 
    {
      NpuRunner runner("AvgPoolV2Grad");
      runner
          .AddInput(x_shape_tensor)
          .AddInput(out_grad_tensor)
          .AddOutput(x_grad_tensor)
          // .SetAttr("orig_input_shape", x_shape)
          .SetAttr("ksize", ksize)   // dims must be 4
          .SetAttr("strides", strides) // dims must be 4
          .SetAttr("padding_mode", "CALCULATED")
          .SetAttr("pads", pads)
          .SetAttr("data_format", "NCHW")
          .SetAttr("global_pooling", global_pool)
          .SetAttr("ceil_mode", false)
          .SetAttr("exclusive", true)
          .Run();
    }  
    x_grad_tensor.print();
#endif
  }
#if 0
  {
    std::swap(x_shape[1], x_shape[3]);
    std::swap(out_shape[1], out_shape[3]);
    NpuTensor<float> x_tensor(x_shape, x_data, ACL_FORMAT_NHWC);
    NpuTensor<float> out_tensor(out_shape, ACL_FORMAT_NHWC);
    {
      NpuRunner runner("AvgPoolV2");
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
          .SetAttr("ksize", {1, 2, 2, 1}) // nchw
          .SetAttr("strides", {1, 2, 2, 1})
          .SetAttr("padding_mode", "CALCULATED")
          .SetAttr("pads", {0, 0, 0, 0})
          .SetAttr("data_format", "NHWC")  // data-format of ksize and strides.
          .SetAttr("global_pooling", false)
          .SetAttr("ceil_mode", false)
          .SetAttr("exclusive", true)
          .Run();
    }
    out_tensor.print();
  }
#endif
  NpuHelper::ReleaseAllDevices();
  return 0;
}
