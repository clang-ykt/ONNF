//===- onnx_ops.cpp - MLIR ONNX Operations --------------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file defines ONNX operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"

#include "onnx_ops.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;

//===----------------------------------------------------------------------===//
// ONNXOpsDialect
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
ONNXOpsDialect::ONNXOpsDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx) {
  addOperations<
#define GET_OP_LIST
#include "src/onnx.cpp.inc"
      >();
}

void ONNXEntryPointOp::build(mlir::Builder *builder,
                             mlir::OperationState &state, mlir::FuncOp function,
                             int numInputs, int numOutputs) {
  state.addAttribute(ONNXEntryPointOp::getEntryPointFuncAttrName(),
                     builder->getSymbolRefAttr(function));
  state.addAttribute(ONNXEntryPointOp::getNumInputsAttrName(),
                     builder->getI32IntegerAttr(numInputs));
  state.addAttribute(ONNXEntryPointOp::getNumOutputsAttrName(),
                     builder->getI32IntegerAttr(numOutputs));
}

ONNXEntryPointOp ONNXEntryPointOp::create(mlir::Location location,
                                          mlir::FuncOp &func, int numInputs,
                                          int numOutputs) {
  mlir::OperationState state(location, "onnx.EntryPoint");
  Builder builder(location->getContext());
  mlir::ONNXEntryPointOp::build(&builder, state, func, numInputs, numOutputs);
  Operation *op = mlir::Operation::create(state);
  auto onnxEntryOp = llvm::cast<mlir::ONNXEntryPointOp>(op);
  return onnxEntryOp;
}

//===----------------------------------------------------------------------===//
// ONNX Operations
//===----------------------------------------------------------------------===//
// Exp
/// Infer the output shape of the ONNXExpOp. This method is required by the
/// shape inference interface.
void ONNXExpOp::inferShapes() { getResult().setType(getOperand().getType()); }

//===----------------------------------------------------------------------===//
// Tanh
/// Infer the output shape of the ONNXTanhOp. This method is required by the
/// shape inference interface.
void ONNXTanhOp::inferShapes() {
  getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// Sinh
/// Infer the output shape of the ONNXSinhOp. This method is required by the
/// shape inference interface.
void ONNXSinhOp::inferShapes() {
  getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// Cosh
/// Infer the output shape of the ONNXCoshOp. This method is required by the
/// shape inference interface.
void ONNXCoshOp::inferShapes() {
  getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// Cos
/// Infer the output shape of the ONNXCosOp. This method is required by the
/// shape inference interface.
void ONNXCosOp::inferShapes() { getResult().setType(getOperand().getType()); }

//===----------------------------------------------------------------------===//
// Log
/// Infer the output shape of the ONNXLogOp. This method is required by the
/// shape inference interface.
void ONNXLogOp::inferShapes() { getResult().setType(getOperand().getType()); }

//===----------------------------------------------------------------------===//
// HardSigmoid
/// Infer the output shape of the ONNXHardSigmoidOp. This method is required by
/// the shape inference interface.
void ONNXHardSigmoidOp::inferShapes() {
  getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// Sigmoid
/// Infer the output shape of the ONNXSigmoidOp. This method is required by the
/// shape inference interface.
void ONNXSigmoidOp::inferShapes() {
  getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// Elu
/// Infer the output shape of the ONNXEluOp. This method is required by the
/// shape inference interface.
void ONNXEluOp::inferShapes() { getResult().setType(getOperand().getType()); }

//===----------------------------------------------------------------------===//
// Relu
/// Infer the output shape of the ONNXReluOp. This method is required by the
/// shape inference interface.
void ONNXReluOp::inferShapes() {
  getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// LeakyRelu
/// Infer the output shape of the ONNXLeakyReluOp. This method is required by
/// the shape inference interface.
void ONNXLeakyReluOp::inferShapes() {
  getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// Selu
/// Infer the output shape of the ONNXSeluOp. This method is required by
/// the shape inference interface.
void ONNXSeluOp::inferShapes() {
  getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// Reciprocal
/// Infer the output shape of the ONNXReciprocalOp. This method is required by
/// the shape inference interface.
void ONNXReciprocalOp::inferShapes() {
  getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// Softmax
/// Infer the output shape of the ONNXSoftmaxOp. This method is required by
/// the shape inference interface.
void ONNXSoftmaxOp::inferShapes() {
  getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//
// Add
/// Infer the output shape of the ONNXAddOp. This method is required by the
/// shape inference interface.
void ONNXAddOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
}

//===----------------------------------------------------------------------===//
// Mul
/// Infer the output shape of the ONNXMulOp. This method is required by the
/// shape inference interface.
void ONNXMulOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
}

//===----------------------------------------------------------------------===//
// Div
/// Infer the output shape of the ONNXDivOp. This method is required by the
/// shape inference interface.
void ONNXDivOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
}

//===----------------------------------------------------------------------===//
// Sub
/// Infer the output shape of the ONNXSubOp. This method is required by the
/// shape inference interface.
void ONNXSubOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
}

//===----------------------------------------------------------------------===//
// And
/// Infer the output shape of the ONNXAndOp. This method is required by the
/// shape inference interface.
void ONNXAndOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
}

//===----------------------------------------------------------------------===//
// Or
/// Infer the output shape of the ONNXOrOp. This method is required by the
/// shape inference interface.
void ONNXOrOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
}

//===----------------------------------------------------------------------===//
// Xor
/// Infer the output shape of the ONNXXorOp. This method is required by the
/// shape inference interface.
void ONNXXorOp::inferShapes() {
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  getResult().setType(getBroadcastedType(lhsTy, rhsTy));
}

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Sum
/// Infer the output shape of the ONNXSumOp. This method is required by the
/// shape inference interface.
void ONNXSumOp::inferShapes() {
  for (int i = 0; i < getNumOperands(); ++i) {
    if (!getOperand(i).getType().cast<RankedTensorType>())
      return;
  }
  Type resultTy = getOperand(0).getType().cast<RankedTensorType>();
  for (int i = 1; i < getNumOperands(); ++i) {
    Type nextTy = getOperand(i).getType().cast<RankedTensorType>();
    resultTy = getBroadcastedType(resultTy, nextTy);
  }
  getResult().setType(resultTy);
}

//===----------------------------------------------------------------------===//
// Max
/// Infer the output shape of the ONNXMaxOp. This method is required by the
/// shape inference interface.
void ONNXMaxOp::inferShapes() {
  for (int i = 0; i < getNumOperands(); ++i) {
    if (!getOperand(i).getType().cast<RankedTensorType>())
      return;
  }
  Type resultTy = getOperand(0).getType().cast<RankedTensorType>();
  for (int i = 1; i < getNumOperands(); ++i) {
    Type nextTy = getOperand(i).getType().cast<RankedTensorType>();
    resultTy = getBroadcastedType(resultTy, nextTy);
  }
  getResult().setType(resultTy);
}

//===----------------------------------------------------------------------===//
// Min
/// Infer the output shape of the ONNXMinOp. This method is required by the
/// shape inference interface.
void ONNXMinOp::inferShapes() {
  for (int i = 0; i < getNumOperands(); ++i) {
    if (!getOperand(i).getType().cast<RankedTensorType>())
      return;
  }
  Type resultTy = getOperand(0).getType().cast<RankedTensorType>();
  for (int i = 1; i < getNumOperands(); ++i) {
    Type nextTy = getOperand(i).getType().cast<RankedTensorType>();
    resultTy = getBroadcastedType(resultTy, nextTy);
  }
  getResult().setType(resultTy);
}

//===----------------------------------------------------------------------===//
// Identity
/// Infer the output shape of the ONNXIdentityOp. This method is required by the
/// shape inference interface.
void ONNXIdentityOp::inferShapes() {
  getResult().setType(getOperand().getType());
}

//===----------------------------------------------------------------------===//

// MatMul

void ONNXMatMulOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  SmallVector<int64_t, 2> dims;
  dims.emplace_back(lhsTy.getShape()[0]);
  dims.emplace_back(rhsTy.getShape()[1]);
  getResult().setType(RankedTensorType::get(dims, lhsTy.getElementType()));
}

// TODO:
//   Verify that matrix sizes are valid.
//   Take into account the dimensionality of the matrix.

//===----------------------------------------------------------------------===//

// Gemm

void ONNXGemmOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  SmallVector<int64_t, 2> dims;
  dims.emplace_back(lhsTy.getShape()[0]);
  dims.emplace_back(rhsTy.getShape()[1]);
  getResult().setType(RankedTensorType::get(dims, lhsTy.getElementType()));
}

// GemmNoBias

void ONNXGemmNoBiasOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!getOperand(0).getType().isa<RankedTensorType>() ||
      !getOperand(1).getType().isa<RankedTensorType>())
    return;
  auto lhsTy = getOperand(0).getType().cast<RankedTensorType>();
  auto rhsTy = getOperand(1).getType().cast<RankedTensorType>();
  SmallVector<int64_t, 2> dims;
  dims.emplace_back(lhsTy.getShape()[0]);
  dims.emplace_back(rhsTy.getShape()[1]);
  getResult().setType(RankedTensorType::get(dims, lhsTy.getElementType()));
}

// TODO:
//   Verify that matrix sizes are valid for multiplication and addition.
//   Take into account the dimensionality of the matrix.

//===----------------------------------------------------------------------===//

// Reshape

void ONNXReshapeOp::inferShapes() {
  // Cannot infer shape if no shape tensor is specified.
  if (!getOperand(1).getType().isa<RankedTensorType>())
    emitError("Shape tensor not ranked.");

  auto inputTensorTy = getOperand(0).getType().cast<RankedTensorType>();
  auto shapeTensorTy = getOperand(1).getType().cast<RankedTensorType>();

  // Only rank 1 shape tensors are supported.
  if (shapeTensorTy.getShape().size() != 1)
    emitError("Shape tensor must have rank one.");

  int64_t outputRank = shapeTensorTy.getShape()[0];

  // Shape tensor must have constant shape.
  if (outputRank < 0)
    emitError("Shape tensor must have constant shape.");

  SmallVector<int64_t, 2> dims;
  for (int i = 0; i < outputRank; ++i)
    dims.emplace_back(-1);

  getResult().setType(
      RankedTensorType::get(dims, inputTensorTy.getElementType()));
}

//===----------------------------------------------------------------------===//

// Transpose

void ONNXTransposeOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!getOperand().getType().isa<RankedTensorType>())
    emitError("Shape tensor not ranked.");

  // Naive transposition which handles the default case of
  // reversing the shape of the tensor (similar to numpy.transpose).
  auto arrayTy = getOperand().getType().cast<RankedTensorType>();
  SmallVector<int64_t, 2> dims;

  if (auto permutation = getAttrOfType<ArrayAttr>(
          ONNXTransposeOp::getPermAttrName())) {
    // Perform transposition according to perm attribute.
    for (auto perm : permutation.getValue())
      dims.emplace_back(arrayTy.getShape()[perm.cast<IntegerAttr>().getInt()]);
  } else {
    // Default
    for (auto dim : llvm::reverse(arrayTy.getShape()))
      dims.emplace_back(dim);
  }

  getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
}

LogicalResult verify(ONNXTransposeOp op) {
  auto module = op.getParentOfType<ModuleOp>();
  if (!module)
    op.emitError("Expected to belong to a module.");

  if (auto permutation = op.getAttrOfType<ArrayAttr>(
          ONNXTransposeOp::getPermAttrName())) {
    for (auto perm : permutation.getValue())
      if (perm.cast<IntegerAttr>().getInt() < 0)
        op.emitError("Cannot tranpose, permuation contains negative index.");
  }

  return success();
}

//===----------------------------------------------------------------------===//

// MaxPool

void ONNXMaxPoolOp::inferShapes() {
  // Cannot infer shape if no shape exists.
  if (!getOperand(0).getType().isa<RankedTensorType>())
    emitError("Shape tensor not ranked.");

  // 1) get shape of input
  auto xTy = getOperand(0).getType().cast<RankedTensorType>();
  auto xShape = xTy.getShape();
  auto xSize = xShape.size();

  // 2) analyse parameters
  // get kernel sizes from kernel_shape attribute 
  auto kernelTy = getAttrOfType<ArrayAttr>(
        ONNXMaxPoolOp::getKernelShapeAttrName());
  if (!kernelTy)
    emitError("kernel_shape is a mandatory attribute.");
  auto kernelSize = kernelTy.getValue().size(); 
  if (kernelSize > xSize)
    emitError("kernel_shape spacial dimension is too large.");

  // now try to find padding, getting auto_pad attribute first
  auto autoPad = getAttrOfType<StringAttr>(
    ONNXMaxPoolOp::getAutoPadAttrName()).getValue();
  // and then investigate the various different cases
  SmallVector<int64_t, 4> actualPads;
  auto defaultPads = false;
  if (autoPad == "NOTSET") {
    if (auto pads = getAttrOfType<ArrayAttr>(
             ONNXConvOp::getPadsAttrName())) {
      // pads consists of two entries for each spatial axis.
      if (pads.getValue().size() != 2 * kernelSize)
        emitError("pads size is not twice the spatial size.");
      // fill in the actual values
      for (int i = 0; i < 2*kernelSize; ++i) {
        // Padding for beginning of axis.
        int64_t p = (pads.getValue()[i]).cast<IntegerAttr>().getInt();
        if (p < 0) 
          emitError("pads value must be nonnegative.");
        actualPads.emplace_back(p);
      }
    } else {
      // pads are not defined, default to value 0
      defaultPads = true;
    }
  } else if (autoPad == "VALID") {
    defaultPads = true;
  } else {
    emitError("auto_pad of unknown / unsupported value.");
  }
  // handle case where default pad values must be used
  if (defaultPads) {
    for(int i=0; i<2*kernelSize; ++i) {
      derivedPads.emplace_back(0)
    }
  }
  // ceil mode
  auto ceilMode = false;
  // dilatation
  // storage order
  // strides

  // initialize output shape 
  SmallVector<int64_t, 4> yShape(xShape);
  // for all kernel dimensions
  for(int i=0; i<kernelSize; ++i) {
    auto inputSpacialShape = xShape[i];
    auto padShape = actualPads[i] + actualPads[kernelSize+i];
    auto kernelSpacialShape = (kernelTy.getValue()[i]).cast<IntegerAttr>().getInt();
    auto dilations = 1;
    auto strideSpacialShape = 1;
    ///output_spatial_shape[i] = ceil( (input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
    double nominator = inputSpacialShape + padShape - 
      ((kernelSpacialShape -1) * dilations + 1;
    double denominator = strideSpacialShape;
    int64_t res = (ceilMode ? ceil(nominator / denominator) : 
      floor(nominator / denominator)) + 1;
    yShape[xSize - kernelSize + i] = res;
  }
  auto arrayTy = getOperand(0).getType().cast<RankedTensorType>();
  getResult(0).setType(RankedTensorType::get(yShape, arrayTy.getElementType()));
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/onnx.cpp.inc"
