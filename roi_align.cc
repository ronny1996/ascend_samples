#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    // NpuHelper::Profiler prof("/work/npu_prof/");
    {
      NpuTensor<float> features({1, 3, 2, 2}, {0.6964692, 0.28613934, 0.22685145, 0.5513148, 0.71946895, 0.42310646, 0.9807642,  0.6848297, 0.4809319, 0.39211753, 0.343178, 0.7290497});
      NpuTensor<float> rois({1, 5}, {0, 1, 0, 3, 3}); // x0, y0, x1, y1
      // NpuTensor<int>   rois_n({1, 5}, {0, 1, 0, 3, 3});

      // bin_w = 3 - 1 = 2
      // bin_h = 3 - 0 = 3
      // grid_c_x1 = (1 + 0.5 * 2 / 2) = 1.5
      // grid_c_y1 = (0 + 0.5 * 3 / 2) = 0.75
      // grid_c_x2 = (3 - 0.5 * 2 / 2) = 2.5
      // grid_c_y2 = 2.25
      // new_boxes = [1.5, 0.75, 2.5, 2.25]
      // crop_size = [2 * 2, 2 * 2] = [4, 4]

      NpuTensor<float> out_tensor({1, 3, 2, 2});
      {
        NpuRunner runner("ROIAlign");
        runner.AddInput(features)
            .AddInput(rois)
            .AddOutput(out_tensor)
            .SetAttr("spatial_scale", 0.5f)
            .SetAttr("pooled_height", 2)
            .SetAttr("pooled_width", 2)
            .SetAttr("sample_num", 2)
            .SetAttr("roi_end_mode", 0)
            .Run();
      }
      out_tensor.print();
      // xxxxx
      // yyyyy
      
      // xy
      // xy
      // xy
      // xy
      // xy

      NpuTensor<float> ydiff({1, 3, 2, 2}, {0.0833333, 0.0833333, 0.0833333, 0.0833333, 0.0833333, 0.0833333, 0.0833333, 0.0833333, 0.0833333, 0.0833333, 0.0833333, 0.0833333});
      NpuTensor<float> xdiff({1, 3, 2, 2});
      NpuTensor<float> rois_d({1, 5}, {0, 1, 3, 0, 3}); // x0, x1, y0, y1
      {
        NpuRunner runner("ROIAlignGrad");
        runner.AddInput(ydiff)
            .AddInput(rois_d)
            .AddOutput(xdiff)
            .SetAttr("xdiff_shape", {1, 3, 2, 2})
            .SetAttr("spatial_scale", 0.5f)
            .SetAttr("pooled_height", 2)
            .SetAttr("pooled_width", 2)
            .SetAttr("sample_num", 2)
            .Run();
      }
      xdiff.print();
    }
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}


// 2 x 4
// x0, y0, x0, y0
// x1, y1, x1, y1
//
// 2 x 2 x 2
//
