#include "lp_norm_grad.h"
#include <vector>

namespace ge {

IMPLEMT_COMMON_INFERFUNC(LpNormGradInferShape) {
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(LpNormGrad, LpNormGradVerify) {
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LpNormGrad, LpNormGradInferShape);

VERIFY_FUNC_REG(LpNormGrad, LpNormGradVerify);

}  // namespace ge
