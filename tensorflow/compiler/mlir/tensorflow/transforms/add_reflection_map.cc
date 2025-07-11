/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir {
namespace emitc {
#define GEN_PASS_DEF_ADDREFLECTIONMAPPASS
#include "mlir/Dialect/EmitC/Transforms/Passes.h.inc"
}  // namespace emitc
}  // namespace mlir

namespace {
class AddReflectionMapPass
    : public impl::AddReflectionMapPassBase<AddReflectionMapPass> {
  void runOnOperation() final;
};

void AddReflectionMapPass::runOnOperation() {
  emitc::ClassOp classOp = getOperation();
  OpBuilder builder(classOp);

  // Create the getBufferForName function
  auto stringViewType =
      emitc::OpaqueType::get(builder.getContext(), "std::string_view");
  auto charPtrType = emitc::OpaqueType::get(builder.getContext(), "char*");
  auto mapType = emitc::OpaqueType::get(builder.getContext(),
                                        "std::map<std::string, char*>");

  // Create function type for getBufferForName
  auto funcType = builder.getFunctionType({stringViewType}, {charPtrType});

  // Insert the function at the end of the class, before the execute function
  auto executeFunc = classOp.lookupSymbol<emitc::FuncOp>("execute");
  builder.setInsertionPoint(executeFunc);

  // Create the getBufferForName function
  auto getBufferFunc = builder.create<emitc::FuncOp>(
      classOp.getLoc(), "getBufferForName", funcType);

  // Add the parameter
  getBufferFunc.insertArgument(0, stringViewType, {}, classOp.getLoc());

  // Create the function body
  Block* funcBody = getBufferFunc.addEntryBlock();
  builder.setInsertionPointToStart(funcBody);

  // Collect all field names
  SmallVector<std::string> fieldNames;

  // Walk through all field operations in the class
  classOp.walk([&](emitc::FieldOp fieldOp) {
    // Get the tf_saved_model.index_path attribute (it's an ArrayAttr, not
    // OpaqueAttr)
    if (auto indexPathAttr =
            fieldOp->getAttrOfType<ArrayAttr>("tf_saved_model.index_path")) {
      if (!indexPathAttr.empty()) {
        if (auto stringAttr = indexPathAttr[0].dyn_cast<StringAttr>()) {
          std::string indexPath = stringAttr.getValue().str();
          fieldNames.push_back(indexPath);
        }
      }
    }
  });

  std::string mapInitializer = "{";
  for (size_t i = 0; i < fieldNames.size(); ++i) {
    if (i > 0) mapInitializer += ", ";
    mapInitializer += "\"" + fieldNames[i] + "\", " +
                      "reinterpret_cast<char*>(&" + fieldNames[i] + ")";
    mapInitializer += "}";
    if (i < fieldNames.size() - 1) mapInitializer += ", {";
  }
  mapInitializer += "}";

  // Create the constant map
  auto mapConstant = builder.create<emitc::ConstantOp>(
      classOp.getLoc(), mapType,
      emitc::OpaqueAttr::get(builder.getContext(), mapInitializer));

  // Return a nullptr
  auto nullConstant = builder.create<emitc::ConstantOp>(
      classOp.getLoc(), charPtrType,
      emitc::OpaqueAttr::get(builder.getContext(), "nullptr"));

  builder.create<emitc::ReturnOp>(classOp.getLoc(), nullConstant.getResult());
}

}  // namespace

std::unique_ptr<OperationPass<emitc::ClassOp>> CreateAddReflectionMapPass() {
  return std::make_unique<AddReflectionMapPass>();
}
