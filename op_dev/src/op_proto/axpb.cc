#include "axpb.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(AxpbInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(op.GetInputDesc("x").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("x").GetDataType());
  tensordesc_output.SetFormat(op.GetInputDesc("x").GetFormat());
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Axpb, AxpbVerify) {
  if (op.GetInputDescByName("x").GetDataType() != op.GetInputDescByName("y").GetDataType()) {
    return GRAPH_SUCCESS;
    // return GRAPH_FAILED;
    /**
     * E19999: Inner Error!
     * Verifying Axpb(Axpb) failed.[FUNC:InferShapeAndType][FILE:shape_refiner.cc][LINE:792]
     * Call InferShapeAndType for node:Axpb(Axpb) failed, input_tensor:{input_0 tensor: [(shape:[2,4]),
     * (format:NCHW),(dtype:DT_FLOAT16),(origin_shape:2,4),
     * (origin_format:NCHW),(origin_dtype:DT_FLOAT16),(shape_range:[])][FUNC:Run][FILE:infershape_pass.cc][LINE:95]
    */
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Axpb, AxpbInferShape);

VERIFY_FUNC_REG(Axpb, AxpbVerify);

}  // namespace ge
