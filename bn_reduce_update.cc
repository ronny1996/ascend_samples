#include "npu_runner.h"

int main(int argc, char const *argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    // NpuHelper::Profiler profiler("./npu_prof/");
    {
      for (auto i = 0; i < 1; i++) {
        // std::cout << i << std::endl;

        std::vector<int64_t> x_shape({1, 2, 2, 2});
        std::vector<float> x_data, y_grad_data;
        for (auto i = 0; i < CommonHelper::Product(x_shape); i++) {
          // x_data.push_back(rand() % 10000 * 1.0 / 10000 - 0.5);
          x_data.push_back(i);
          // y_grad_data.push_back(rand() % 10000 * 1.0 / 10000 - 0.5);
          y_grad_data.push_back(1);
        }

        NpuTensor<float> x_tensor({x_shape}, x_data);
        NpuTensor<float> x_sum({x_shape[1]});
        NpuTensor<float> x_square_sum({x_shape[1]});


        NpuTensor<float> scale_tensor(
            {x_shape[1]},
            std::vector<float>(x_shape[1], 1));
        NpuTensor<float> offset_tensor(
            {x_shape[1]},
            std::vector<float>(x_shape[1], 0));
        NpuTensor<float> mean_tensor(
            {x_shape[1]},
            std::vector<float>(x_shape[1], 0));
        NpuTensor<float> variance_tensor(
            {x_shape[1]},
            std::vector<float>(x_shape[1], 1));

        NpuTensor<float> out_tensor({x_shape});
        NpuTensor<float> batch_mean(
            {x_shape[1]},
            std::vector<float>(x_shape[1], 0));
        NpuTensor<float> batch_variance(
            {x_shape[1]},
            std::vector<float>(x_shape[1], 1));
        NpuTensor<float> reserve_space_1(
            {x_shape[1]},
            std::vector<float>(x_shape[1], 0));
        NpuTensor<float> reserve_space_2(
            {x_shape[1]},
            std::vector<float>(x_shape[1], 1));
        {
          NpuRunner runner("BNTrainingReduce");
          runner.AddInput(x_tensor)
              .AddOutput(x_sum)
              .AddOutput(x_square_sum)
              .Run();
        }
        x_sum.print();
        x_square_sum.print();
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
        out_tensor.print();
        reserve_space_1.print(); // mean * (1 - factor) + factor * batch_mean
        reserve_space_2.print(); // var * (1 - factor) + factor * batch_var * N / (N - 1)
        batch_mean.print();     // 1.5, 5.5
        batch_variance.print(); // 1.25, 1.25
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
              .SetAttr("epsilon", static_cast<float>(1e-7))
              .SetAttr("data_format", "NCHW")
              .SetAttr("is_training", true)
              .SetAttr("factor", 0.8f)
              .Run();
        }
        out_tensor.print();
        reserve_space_1.print(); // mean * (1 - factor) + factor * batch_mean
        reserve_space_2.print(); // var * (1 - factor) + factor * batch_var * N / (N - 1)
        batch_mean.print();     // 1.5, 5.5
        batch_variance.print(); // 1.25, 1.25

        NpuTensor<float> y_grad_tensor(x_shape, y_grad_data);
        NpuTensor<float> x_grad_tensor(x_shape);
        NpuTensor<float> scale_grad_tensor({x_shape[1]});
        NpuTensor<float> offset_grad_tensor({x_shape[1]});
        NpuTensor<float> reserve_space_3({x_shape[1]});
        NpuTensor<float> reserve_space_4({x_shape[1]});
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
        // {
        //   NpuRunner runner("BatchNormGrad");
        //   runner.AddInput(y_grad_tensor)
        //       .AddInput(x_tensor)
        //       .AddInput(scale_tensor)
        //       .AddInput(batch_mean)
        //       .AddInput(batch_variance)
        //       .AddOutput(x_grad_tensor)
        //       .AddOutput(scale_grad_tensor)
        //       .AddOutput(offset_grad_tensor)
        //       .AddOutput(reserve_space_3)
        //       .AddOutput(reserve_space_4)
        //       .SetAttr("data_format", "NCHW")
        //       .SetAttr("epsilon", 1e-7f)
        //       .SetAttr("is_training", true)
        //       .Run();
        // }
        x_grad_tensor.print();
        scale_grad_tensor.print();
        offset_grad_tensor.print();
      }
    }
  }

  NpuHelper::ReleaseAllDevices();
  return 0;
}
