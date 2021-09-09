#include "npu_runner.h"

template <typename T>
void batch_norm_grad(NpuTensor<T> &y_grad, NpuTensor<T> &x, NpuTensor<T> &mean,
                     NpuTensor<T> &var, NpuTensor<T> &scale, float epsilon,
                     NpuTensor<T> &x_grad, NpuTensor<T> &scale_grad,
                     NpuTensor<T> &bias_grad) {
  auto trans_op = [](NpuTensor<T> &src, NpuTensor<T> &dst,
                     const std::string &src_fmt, const std::string &dst_fmt) {
    NpuRunner runner("TransData");
    runner.AddInput(src)
        .AddOutput(dst)
        .SetAttr("src_format", src_fmt.c_str())
        .SetAttr("dst_format", dst_fmt.c_str())
        .Run();
  };

  NpuTensor<T> norm_x(CommonHelper::TransToChannelLast(x.dims),
                      ACL_FORMAT_NHWC);
  NpuTensor<T> norm_x_grad(CommonHelper::TransToChannelLast(x.dims),
                           ACL_FORMAT_NHWC);
  NpuTensor<T> y_grad_nchw(CommonHelper::TransToChannelLast(y_grad.dims),
                           ACL_FORMAT_NHWC);
  NpuTensor<T> std_var(var.dims);

  trans_op(y_grad, y_grad_nchw, "NCHW", "NHWC");
  trans_op(x, norm_x, "NCHW", "NHWC");
  {
    // get norm_x
    {
      NpuRunner runner1("Adds");
      runner1.AddInput(var).AddOutput(std_var).SetAttr("value", epsilon).Run();
      NpuRunner runner2("Sqrt");
      runner2.AddInput(std_var).AddOutput(std_var).Run();
      NpuRunner runner3("Sub");
      runner3.AddInput(norm_x).AddInput(mean).AddOutput(norm_x).Run();
      NpuRunner runner4("Div");
      runner4.AddInput(norm_x).AddInput(std_var).AddOutput(norm_x).Run();
    }
    // get scale_grad, bias_grad
    {
      NpuTensor<T> y_grad_mul_norm_x(CommonHelper::TransToChannelLast(x.dims),
                                     ACL_FORMAT_NHWC);
      NpuRunner runner1("Mul");
      runner1.AddInput(y_grad_nchw)
          .AddInput(norm_x)
          .AddOutput(y_grad_mul_norm_x)
          .Run();
      NpuRunner runner2("ReduceSumD");
      runner2.AddInput(y_grad_mul_norm_x)
          .AddOutput(scale_grad)
          .SetAttr("keep_dims", false)
          .SetAttr("axes", {0, 1, 2})
          .Run();
      NpuRunner runner3("ReduceSumD");
      runner3.AddInput(y_grad_nchw)
          .AddOutput(bias_grad)
          .SetAttr("keep_dims", false)
          .SetAttr("axes", {0, 1, 2})
          .Run();
    }
    // get x_grad
    {
      NpuTensor<T> y_grad_mul_N(CommonHelper::TransToChannelLast(x.dims),
                                ACL_FORMAT_NHWC);
      NpuTensor<T> norm_x_mul_scale_grad(
          CommonHelper::TransToChannelLast(x.dims), ACL_FORMAT_NHWC);
      NpuTensor<T> scale_div_std_var(var.dims);

      NpuRunner runner1("Muls");
      auto &dims = y_grad_mul_N.dims;
      runner1.AddInput(y_grad_nchw)
          .AddOutput(y_grad_mul_N)
          .SetAttr("value", static_cast<float>(dims[0] * dims[1] * dims[2]))
          .Run();
      NpuRunner runner2("Sub");
      runner2.AddInput(y_grad_mul_N)
          .AddInput(bias_grad)
          .AddOutput(y_grad_mul_N)
          .Run();
      NpuRunner runner3("Mul");
      runner3.AddInput(norm_x)
          .AddInput(scale_grad)
          .AddOutput(norm_x_mul_scale_grad)
          .Run();
      NpuRunner runner4("Sub");
      runner4.AddInput(y_grad_mul_N)
          .AddInput(norm_x_mul_scale_grad)
          .AddOutput(y_grad_mul_N)
          .Run();
      NpuRunner runner5("Div");
      runner5.AddInput(scale)
          .AddInput(std_var)
          .AddOutput(scale_div_std_var)
          .Run();
      NpuRunner runner6("Mul");
      runner6.AddInput(y_grad_mul_N)
          .AddInput(scale_div_std_var)
          .AddOutput(y_grad_mul_N)
          .Run();
      NpuRunner runner7("Muls");
      runner7.AddInput(y_grad_mul_N)
          .AddOutput(y_grad_mul_N)
          .SetAttr("value",
                   static_cast<float>(1.0 / (dims[0] * dims[1] * dims[2])))
          .Run();
      trans_op(y_grad_mul_N, x_grad, "NHWC", "NCHW");
    }
  }
}

int main(int argc, char const *argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    // NpuHelper::Profiler profiler("./npu_prof/");
    {
      for (auto i = 0; i < 10; i++) {
        std::cout << i << std::endl;

        std::vector<int64_t> x_shape({3, 32, 112, 112});
        std::vector<float> x_data, y_grad_data;
        for (auto i = 0; i < CommonHelper::Product(x_shape); i++) {
          x_data.push_back(rand() % 10000 * 1.0 / 10000 - 0.5);
          y_grad_data.push_back(rand() % 10000 * 1.0 / 10000 - 0.5);
        }

        NpuTensor<float> x_tensor({x_shape}, x_data);
        NpuTensor<float> x_sum({x_shape[1]});
        NpuTensor<float> x_square_sum({x_shape[1]});

        NpuTensor<float> scale_tensor(
            {x_shape[1]},
            std::vector<float>(x_shape[1], rand() % 10000 / 10000 - 0.5));
        NpuTensor<float> offset_tensor(
            {x_shape[1]},
            std::vector<float>(x_shape[1], rand() % 10000 / 10000 - 0.5));
        NpuTensor<float> mean_tensor(
            {x_shape[1]},
            std::vector<float>(x_shape[1], rand() % 10000 / 10000 - 0.5));
        NpuTensor<float> variance_tensor(
            {x_shape[1]},
            std::vector<float>(x_shape[1], rand() % 10000 / 10000 - 0.5));

        NpuTensor<float> out_tensor({x_shape});
        NpuTensor<float> batch_mean(
            {x_shape[1]},
            std::vector<float>(x_shape[1], rand() % 10000 / 10000 - 0.5));
        NpuTensor<float> batch_variance(
            {x_shape[1]},
            std::vector<float>(x_shape[1], rand() % 10000 / 10000 - 0.5));
        NpuTensor<float> reserve_space_1(
            {x_shape[1]},
            std::vector<float>(x_shape[1], rand() % 10000 / 10000 - 0.5));
        NpuTensor<float> reserve_space_2(
            {x_shape[1]},
            std::vector<float>(x_shape[1], rand() % 10000 / 10000 - 0.5));

        {
          NpuRunner runner("BNTrainingReduce");
          runner.AddInput(x_tensor)
              .AddOutput(x_sum)
              .AddOutput(x_square_sum)
              .Run();
        }
        // x_sum.print();
        // x_square_sum.print();
        {
          NpuRunner runner("BNTrainingUpdate");
          runner.AddInput(x_tensor)
              .AddInput(x_sum)
              .AddInput(x_square_sum)
              .AddInput(scale_tensor)
              .AddInput(offset_tensor)
              .AddInput(mean_tensor)  
              .AddInput(variance_tensor) 
              .AddOutput(out_tensor)
              .AddOutput(reserve_space_1)  // mean_for_grad
              .AddOutput(reserve_space_2)  // variance_for_grad
              .AddOutput(batch_mean)
              .AddOutput(batch_variance)
              .SetAttr("epsilon", static_cast<float>(1e-7))
              .SetAttr("factor", 0.8f)
              .Run();
        }
        {
          NpuRunner runner("BatchNorm");
          runner.AddInput(x_tensor)
              .AddInput(scale_tensor)
              .AddInput(offset_tensor)
              .AddInput(mean_tensor)  // Must be"None" if the operation is used
                                      // for training
              .AddInput(variance_tensor)  // Must be"None" if the operation is
                                          // used for training
              .AddOutput(out_tensor)
              .AddOutput(reserve_space_1)  // mean_for_grad
              .AddOutput(reserve_space_2)  // variance_for_grad
              .AddOutput(batch_mean)
              .AddOutput(batch_variance)
              .SetAttr("epsilon", static_cast<float>(1e-4))
              .SetAttr("data_format", "NCHW")
              .SetAttr("is_training", true)
              .Run();
        }
        // out_tensor.print();
        // batch_mean.print();
        // batch_variance.print();
        // reserve_space_1.print();
        // reserve_space_2.print();

        NpuTensor<float> y_grad_tensor(x_shape, y_grad_data);
        NpuTensor<float> x_grad_tensor(x_shape);
        NpuTensor<float> scale_grad_tensor({x_shape[1]});
        NpuTensor<float> offset_grad_tensor({x_shape[1]});
        NpuTensor<float> reserve_space_3({x_shape[1]});
        NpuTensor<float> reserve_space_4({x_shape[1]});

        {
          NpuRunner runner("BatchNormGrad");
          runner.AddInput(y_grad_tensor)
              .AddInput(x_tensor)       // 3, 32, 112, 112
              .AddInput(scale_tensor)   // 32
              .AddInput(batch_mean)     // 32
              .AddInput(batch_variance) // 32
              .AddOutput(x_grad_tensor) // 3, 32, 112, 112
              .AddOutput(scale_grad_tensor)// 32
              .AddOutput(offset_grad_tensor)// 32
              .AddOutput(reserve_space_3)
              .AddOutput(reserve_space_4)
              .SetAttr("data_format", "NCHW")
              .SetAttr("epsilon", 1e-7f)
              .SetAttr("is_training", true)
              .Run();
        }
        x_grad_tensor.sync();
        scale_grad_tensor.sync();
        offset_grad_tensor.sync();
        {
          NpuRunner runner("BNTrainingUpdateGrad");
          runner.AddInput(y_grad_tensor)
                .AddInput(x_tensor)
                .AddInput(batch_mean)
                .AddInput(batch_variance)
                .AddOutput(scale_grad_tensor)
                .AddOutput(offset_grad_tensor)
                .SetAttr("epsilon", 1e-7f)
                .Run();
        }
        {
          NpuRunner runner("BNTrainingReduceGrad");
          runner.AddInput(y_grad_tensor)
                .AddInput(x_tensor)
                .AddInput(scale_grad_tensor)
                .AddInput(offset_grad_tensor)
                .AddInput(scale_tensor)
                .AddInput(batch_mean)
                .AddInput(batch_variance)
                .AddOutput(x_grad_tensor)
                .SetAttr("epsilon", 1e-7f)
                .Run();
        }
        x_grad_tensor.sync();
        scale_grad_tensor.sync();
        offset_grad_tensor.sync();
        // for (auto i = 0; i < 10; i++) {
        //   {
        //     batch_norm_grad(y_grad_tensor, x_tensor, batch_mean,
        //     batch_variance,
        //                     scale_tensor, 1e-7f, x_grad_tensor,
        //                     scale_grad_tensor, offset_grad_tensor);
        //   }
        // }
      }
    }
  }

  NpuHelper::ReleaseAllDevices();
  return 0;
}
