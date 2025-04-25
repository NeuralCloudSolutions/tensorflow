/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/spmd/shardy/round_trip_common/export_named_computations.h"

#include <cassert>
#include <memory>
#include <optional>

#include "absl/log/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::ArrayAttr;
using ::mlir::ModuleOp;
using ::mlir::NamedAttribute;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;

using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::ManualAxesAttr;
using ::mlir::sdy::NamedComputationOp;
using ::mlir::sdy::TensorShardingAttr;
using ::mlir::sdy::TensorShardingPerValueAttr;

// Converts a `NamedComputationOp` into a `CallOp`.
class ExportNamedComputationsPass
    : public mlir::PassWrapper<ExportNamedComputationsPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportNamedComputationsPass)

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);
    mlir::Block& moduleBlock = moduleOp.getRegion().front();
    getOperation()->walk([&](NamedComputationOp namedComputationOp) {
      mlir::IRRewriter rewriter(namedComputationOp);
      rewriter.setInsertionPointToEnd(&moduleBlock);
      auto funcOp = rewriter.create<FuncOp>(
          namedComputationOp.getLoc(), namedComputationOp.getName(),
          rewriter.getFunctionType(
              namedComputationOp.getBody().getArgumentTypes(),
              namedComputationOp.getResultTypes()),
          rewriter.getStringAttr("private"),
          /*argAttrs=*/ArrayAttr(), /*resultAttrs=*/ArrayAttr());
      rewriter.setInsertionPointToStart(funcOp->getBlock());
      mlir::sdy::inlineRegionAndConvertTerminatorOp<mlir::func::ReturnOp>(
          namedComputationOp.getBody(), funcOp.getBody());
      rewriter.setInsertionPoint(namedComputationOp);

      // `namedComputationOp` has manual axes, implying that it is in a manual
      // computation. We export the shardings to `kXlaShardingAttr`.
      if (ManualAxesAttr manualAxes =
              namedComputationOp->getAttrOfType<ManualAxesAttr>(kManualAxes)) {
        CHECK_GT(manualAxes.size(), 0);
        auto getMeshAttr = [&](TensorShardingAttr sharding) {
          return sharding.getMesh(symbolTable);
        };
        mlir::MLIRContext* context = funcOp.getContext();

        std::optional<TensorShardingPerValueAttr> inShardings =
            namedComputationOp.getInShardings();
        std::optional<TensorShardingPerValueAttr> outShardings =
            namedComputationOp.getOutShardings();
        assert(inShardings.has_value());
        assert(outShardings.has_value());

        // Copy the input shardings to the func.
        for (auto [i, sharding] :
             llvm::enumerate(inShardings->getShardings())) {
          HloSharding hloSharding =
              convertToHloSharding(sharding, getMeshAttr, manualAxes);
          funcOp.setArgAttr(
              i, kXlaShardingAttr,
              mlir::StringAttr::get(context, hloSharding.ToString()));
        }

        // Copy the output shardings to the func AND call.
        for (auto [i, sharding] :
             llvm::enumerate(outShardings->getShardings())) {
          HloSharding hloSharding =
              convertToHloSharding(sharding, getMeshAttr, manualAxes);
          funcOp.setResultAttr(
              i, kXlaShardingAttr,
              mlir::StringAttr::get(context, hloSharding.ToString()));
        }

        mlir::StringAttr funcName = symbolTable.insert(funcOp);
        auto callOp = rewriter.replaceOpWithNewOp<CallOp>(
            namedComputationOp, namedComputationOp.getResultTypes(), funcName,
            namedComputationOp.getOperands());
        callOp->setAttr(
            kXlaShardingAttr,
            convertToHloShardingAttr(callOp, outShardings->getShardings(),
                                     getMeshAttr, manualAxes));
        return;
      }

      // `namedComputationOp` does not have manual axes. We export the shardings
      // to `kShardingAttr`. First, copy the input shardings to the func.
      if (std::optional<TensorShardingPerValueAttr> inShardings =
              namedComputationOp.getInShardings()) {
        for (auto [arg, sharding] : llvm::zip_equal(
                 funcOp.getArguments(), inShardings->getShardings())) {
          setSharding(arg, sharding);
        }
      }

      // Copy the output shardings to the func AND call.
      mlir::SmallVector<NamedAttribute> callOpAttrs(
          namedComputationOp->getDiscardableAttrs());
      if (std::optional<TensorShardingPerValueAttr> outShardings =
              namedComputationOp.getOutShardings()) {
        for (auto [i, sharding] :
             llvm::enumerate(outShardings->getShardings())) {
          funcOp.setResultAttr(i, kShardingAttr, sharding);
        }
        callOpAttrs.push_back(NamedAttribute(
            rewriter.getStringAttr(kShardingAttr), *outShardings));
      }

      mlir::StringAttr funcName = symbolTable.insert(funcOp);
      auto callOp = rewriter.replaceOpWithNewOp<CallOp>(
          namedComputationOp, namedComputationOp.getResultTypes(), funcName,
          namedComputationOp.getOperands());
      callOp->setAttrs(callOpAttrs);
    });
  }

  StringRef getArgument() const override {
    return "xla-sdy-export-named-computations";
  }

  StringRef getDescription() const override {
    return "Creates a pass that converts a `NamedComputationOp` with a "
           "`to a `CallOp` with a new private function "
           "called the `NamedComputationOp`'s `name`. The new `FuncOp` and "
           "`CallOp` have the same shardings as the original "
           "`NamedComputationOp`s operands/results.";
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createExportNamedComputationsPass() {
  return std::make_unique<ExportNamedComputationsPass>();
}

void registerExportNamedComputationsPass() {
  mlir::registerPass(createExportNamedComputationsPass);
}

}  // namespace sdy
}  // namespace xla
