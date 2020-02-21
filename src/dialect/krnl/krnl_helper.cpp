#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExpr.h"

#include "src/dialect/krnl/krnl_ops.hpp"

#include "krnl_helper.hpp"

namespace onnf {

using namespace mlir;

ParseResult
KrnlDialectOperandParser::ParseOptionalOperand(const Type &operandType,
                                               Value &operand) {
  // If operand queue is empty, parse more operands and cache them.
  if (_operandRefQueue.empty()) {
    // Parse operand types:
    llvm::SmallVector<OpAsmParser::OperandType, 2> operand_refs;
    _parser.parseOperandList(operand_refs);

    // Record operands:
    for (auto &operand_ref : operand_refs)
      _operandRefQueue.emplace(operand_ref);
  }

  // If we parsed some operand reference(s), resolve the ref to an operand:
  if (!_operandRefQueue.empty()) {
    auto operand_ref = _operandRefQueue.front();
    _operandRefQueue.pop();

    llvm::SmallVector<Value, 1> operands;
    _parser.resolveOperand(operand_ref, operandType, operands);
    operand = operands.front();
    return success();
  } else {
    operand = nullptr;
    return failure();
  }
}

ParseResult KrnlDialectOperandParser::ParseOptionalOperand(
    const Type &operandType, llvm::SmallVectorImpl<Value> &operandList) {
  Value operand = nullptr;
  if (ParseOptionalOperand(operandType, operand))
    return failure();

  operandList.emplace_back(operand);
  return success();
}

ParseResult KrnlDialectOperandParser::ParseOperand(const Type &operandType,
                                                   Value &operand) {
  if (ParseOptionalOperand(operandType, operand))
    return _parser.emitError(_parser.getCurrentLocation(),
                             "Expecting an operand.");
  return success();
}

ParseResult KrnlDialectOperandParser::ParseOperand(
    const Type &operandType, llvm::SmallVectorImpl<Value> &operandList) {
  if (ParseOptionalOperand(operandType, operandList))
    return _parser.emitError(_parser.getCurrentLocation(),
                             "Expecting an operand.");

  return success();
}

void printDimAndSymbolList(Operation::operand_iterator &begin, unsigned numDims,
                           unsigned numSymbols, OpAsmPrinter &p) {
  p << '(';
  p.printOperands(begin, begin + numDims);
  p << ')';

  if (numSymbols) {
    p << '[';
    p.printOperands(begin + numDims, begin + numDims + numSymbols);
    p << ']';
  }

  begin = std::next(begin, numDims + numSymbols);
}

void printBound(AffineMapAttr boundMap,
                Operation::operand_iterator &boundOperandsBeg,
                const char *prefix, OpAsmPrinter &p) {
  AffineMap map = boundMap.getValue();

  // Check if this bound should be printed using custom assembly form.
  // The decision to restrict printing custom assembly form to trivial cases
  // comes from the will to roundtrip MLIR binary -> text -> binary in a
  // lossless way.
  // Therefore, custom assembly form parsing and printing is only supported for
  // zero-operand constant maps and single symbol operand identity maps.
  if (map.getNumResults() == 1) {
    AffineExpr expr = map.getResult(0);

    // Print constant bound.
    if (map.getNumDims() == 0 && map.getNumSymbols() == 0) {
      if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
        p << constExpr.getValue();
        return;
      }
    }

    // Print bound that consists of a single SSA symbol if the map is over a
    // single symbol.
    if (map.getNumDims() == 0 && map.getNumSymbols() == 1) {
      if (auto symExpr = expr.dyn_cast<AffineSymbolExpr>()) {
        p.printOperand(*(boundOperandsBeg++));
        return;
      }
    }
  } else {
    // Map has multiple results. Print 'min' or 'max' prefix.
    p << prefix << ' ';
  }

  // Print the map and its operands.
  p << boundMap;
  printDimAndSymbolList(boundOperandsBeg, map.getNumDims(), map.getNumSymbols(),
                        p);
}
} // namespace onnf

namespace mlir {

void KrnlIterateOperandPack::pushConstantBound(int64_t bound) {
  if (boundMaps.size() % 2 == 0)
    _operands.emplace_back(inputLoops[boundMaps.size() / 2]);
  AffineMap map = builder.getConstantAffineMap(bound);
  boundMaps.emplace_back(AffineMapAttr::get(map));
}

void KrnlIterateOperandPack::pushOperandBound(mlir::Value operand) {
  if (boundMaps.size() % 2 == 0)
    _operands.emplace_back(inputLoops[boundMaps.size() / 2]);
  AffineMap map = builder.getSymbolIdentityMap();
  boundMaps.emplace_back(AffineMapAttr::get(map));
  _operands.emplace_back(operand);
}

BuildKrnlLoop::BuildKrnlLoop(ConversionPatternRewriter &rewriter, Location loc)
    : rewriter(rewriter), loc(loc), originalLoopNum(-1), pack(NULL),
      pushCount(0), createdOptimizeOp(false), createdIterOp(false) {}

BuildKrnlLoop::~BuildKrnlLoop() {
  if (!createdOptimizeOp)
    emitError(loc, "expected to create optimize op");
  if (!createdIterOp)
    emitError(loc, "expected to create iteration op");
  if (pack)
    free(pack);
}

void BuildKrnlLoop::createOptimizeOp(int loopNum, bool withEmptyOptimization) {
  originalLoopNum = loopNum;
  if (originalLoopNum <= 0)
    emitError(loc, "expected positive number of original loops");
  // insert define loop op
  auto loopsOp = rewriter.create<KrnlDefineLoopsOp>(loc, originalLoopNum);
  originalLoops.reserve(originalLoopNum);
  for (auto result : loopsOp.getResults())
    originalLoops.push_back(result);
  // inserte optimize loop op.
  auto optimizedLoopsOp =
      rewriter.create<KrnlOptimizeLoopsOp>(loc, originalLoopNum);
  optLoops.reserve(originalLoopNum);
  // Emit empty optimizations

  if (withEmptyOptimization) {
    for (auto result : optimizedLoopsOp.getResults())
      optLoops.push_back(result);
    optBlock = &optimizedLoopsOp.region().front();
    auto ip = rewriter.saveInsertionPoint();
    insertInOptimizeLoopEnd();
    rewriter.create<KrnlReturnLoopsOp>(loc, originalLoops);
    rewriter.restoreInsertionPoint(ip);
  }
  // prepare data structure to push bounds
  pack = new KrnlIterateOperandPack(rewriter, originalLoops, optLoops);
  createdOptimizeOp = true;
}

// push bounds (lower and upper) and return index for loop info
int BuildKrnlLoop::pushBounds(int64_t lb, int64_t ub) {
  pack->pushConstantBound(lb);
  pack->pushConstantBound(ub);
  return pushCount++;
}

int BuildKrnlLoop::pushBounds(int64_t lb, Value ub) {
  pack->pushConstantBound(lb);
  pack->pushOperandBound(ub);
  return pushCount++;
}

int BuildKrnlLoop::pushBounds(int64_t lb, Value ubMemRefOperand,
                              int ubMemRefIndex, bool ubMustBeConstant) {
  pack->pushConstantBound(lb);
  // process ub as a dimension of mem ref, possibly non-constant
  auto shape = ubMemRefOperand.getType().cast<MemRefType>().getShape();
  if (shape[ubMemRefIndex] < 0) {
    if (ubMustBeConstant)
      emitError(loc, "bound expected to be constant");
    pack->pushOperandBound(
        rewriter.create<DimOp>(loc, ubMemRefOperand, ubMemRefIndex)
            .getResult());
  } else
    pack->pushConstantBound(shape[ubMemRefIndex]);
  return pushCount++;
}

int BuildKrnlLoop::pushBounds(Value lb, Value ub) {
  pack->pushOperandBound(lb);
  pack->pushOperandBound(ub);
  return pushCount++;
}

// create iter
void BuildKrnlLoop::createIterateOp() {
  if (!createdOptimizeOp)
    emitError(loc, "must create optimize op before iterate op");
  // have to have defined all bounds
  if (pushCount != originalLoopNum) {
    printf(" push count %d, original loop %d\n", pushCount, originalLoopNum);
    emitError(loc, "must push bounds for all original loops");
  }
  // create iterate op
  auto iterateOp = rewriter.create<KrnlIterateOp>(loc, *pack);
  iterBlock = &iterateOp.bodyRegion().front();
  createdIterOp = true;
}

void BuildKrnlLoop::createOptimizeAndIterateOp(Value memRefOperand,
                                               bool withEmptyOptimization) {
  int loopNum = memRefOperand.getType().cast<MemRefType>().getShape().size();
  createOptimizeOp(loopNum, withEmptyOptimization);
  for (int i = 0; i < originalLoopNum; ++i)
    pushBounds(0, memRefOperand, i);
  createIterateOp();
}

// get induction variable to be use within iter
BlockArgument &BuildKrnlLoop::getInductionVar(int originalLoopIndex) {
  if (originalLoopIndex < 0 || originalLoopIndex >= originalLoopNum)
    emitError(loc, "original loop index is out of bound");
  return iterBlock->getArguments()[originalLoopIndex];
}

// set insertion points
void BuildKrnlLoop::insertInOptimizeLoopStart() {
  rewriter.setInsertionPointToStart(optBlock);
}

void BuildKrnlLoop::insertInOptimizeLoopEnd() {
  rewriter.setInsertionPointToEnd(optBlock);
}

void BuildKrnlLoop::insertInIterateLoopStart() {
  rewriter.setInsertionPointToStart(iterBlock);
}

void BuildKrnlLoop::insertInIterateLoopEnd() {
  rewriter.setInsertionPointToEnd(iterBlock);
}

} // namespace mlir
