//===- SCFToGPU.h - Convert loop nests to GPU kernels -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_SCFTOGPU_SCFTOGPU_H_
#define MLIR_CONVERSION_SCFTOGPU_SCFTOGPU_H_

#include "mlir/Support/LLVM.h"

namespace mlir {
class AffineForOp;
class MLIRContext;
class OwningRewritePatternList;
struct LogicalResult;
class Value;

namespace scf {
class ForOp;
} // end namespace scf

/// Convert a perfect affine loop nest with the outermost loop identified by
/// `forOp` into a gpu::Launch operation.  Map `numBlockDims` outer loops to
/// GPU blocks and `numThreadDims` to GPU threads.  The bounds of the loops that
/// are mapped should be independent of the induction variables of the other
/// mapped loops.
///
/// No check on the size of the block or grid, or on the validity of
/// parallelization is performed, it is under the responsibility of the caller
/// to strip-mine the loops and to perform the dependence analysis before
/// calling the conversion.

// TODO: Consider removing this in favor of affine.for -> affine.parallel
// detection followed by an affine.parallel -> scf.parallel -> gpu.launch
// conversion
LogicalResult convertAffineLoopNestToGPULaunch(AffineForOp forOp,
                                               unsigned numBlockDims,
                                               unsigned numThreadDims);

/// Adds the conversion pattern from `scf.parallel` to `gpu.launch` to the
/// provided pattern list.
void populateParallelLoopToGPUPatterns(OwningRewritePatternList &patterns,
                                       MLIRContext *ctx);

} // namespace mlir

#endif // MLIR_CONVERSION_SCFTOGPU_SCFTOGPU_H_
