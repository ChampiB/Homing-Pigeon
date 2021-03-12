  def MatmulOp : LinalgStructuredBase_Op<"matmul", [
    NamedStructuredOpTrait,
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"YieldOp">]> {
      let arguments = (ins Variadic<AnyShaped>:$inputs,
                           Variadic<AnyMemRef>:$output_buffers,
                           Variadic<AnyRankedTensor>:$init_tensors);
      let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
      let regions = (region AnyRegion:$region);

      let skipDefaultBuilders = 1;
      let builders = [ OpBuilder<
        "ValueRange inputs, ValueRange outputBuffers",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<MatmulOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange inputs, "
        "ValueRange outputBuffers, ValueRange initTensors",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addOperands(initTensors);
          $_state.addTypes(resultTensorTypes);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<MatmulOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {}",
        [{
          $_state.addOperands(operands);
          $_state.addAttributes(attributes);
          $_state.addTypes(resultTensorTypes);
          (void)$_state.addRegion();
        }]>
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{ return ::parseNamedStructuredOp<MatmulOp>(parser, result); }];
      let verifier = [{ return ::verifyNamedStructuredOp(*this); }];
      let hasFolder = 1;
      let hasCanonicalizer = 1;

      let extraClassDeclaration = [{
        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      }];
  }
  def MatvecOp : LinalgStructuredBase_Op<"matvec", [
    NamedStructuredOpTrait,
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"YieldOp">]> {
      let arguments = (ins Variadic<AnyShaped>:$inputs,
                           Variadic<AnyMemRef>:$output_buffers,
                           Variadic<AnyRankedTensor>:$init_tensors);
      let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
      let regions = (region AnyRegion:$region);

      let skipDefaultBuilders = 1;
      let builders = [ OpBuilder<
        "ValueRange inputs, ValueRange outputBuffers",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<MatvecOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange inputs, "
        "ValueRange outputBuffers, ValueRange initTensors",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addOperands(initTensors);
          $_state.addTypes(resultTensorTypes);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<MatvecOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {}",
        [{
          $_state.addOperands(operands);
          $_state.addAttributes(attributes);
          $_state.addTypes(resultTensorTypes);
          (void)$_state.addRegion();
        }]>
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{ return ::parseNamedStructuredOp<MatvecOp>(parser, result); }];
      let verifier = [{ return ::verifyNamedStructuredOp(*this); }];
      let hasFolder = 1;
      let hasCanonicalizer = 1;

      let extraClassDeclaration = [{
        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      }];
  }
  def VecmatOp : LinalgStructuredBase_Op<"vecmat", [
    NamedStructuredOpTrait,
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"YieldOp">]> {
      let arguments = (ins Variadic<AnyShaped>:$inputs,
                           Variadic<AnyMemRef>:$output_buffers,
                           Variadic<AnyRankedTensor>:$init_tensors);
      let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
      let regions = (region AnyRegion:$region);

      let skipDefaultBuilders = 1;
      let builders = [ OpBuilder<
        "ValueRange inputs, ValueRange outputBuffers",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<VecmatOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange inputs, "
        "ValueRange outputBuffers, ValueRange initTensors",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addOperands(initTensors);
          $_state.addTypes(resultTensorTypes);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<VecmatOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {}",
        [{
          $_state.addOperands(operands);
          $_state.addAttributes(attributes);
          $_state.addTypes(resultTensorTypes);
          (void)$_state.addRegion();
        }]>
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{ return ::parseNamedStructuredOp<VecmatOp>(parser, result); }];
      let verifier = [{ return ::verifyNamedStructuredOp(*this); }];
      let hasFolder = 1;
      let hasCanonicalizer = 1;

      let extraClassDeclaration = [{
        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      }];
  }
  def DotOp : LinalgStructuredBase_Op<"dot", [
    NamedStructuredOpTrait,
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"YieldOp">]> {
      let arguments = (ins Variadic<AnyShaped>:$inputs,
                           Variadic<AnyMemRef>:$output_buffers,
                           Variadic<AnyRankedTensor>:$init_tensors);
      let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
      let regions = (region AnyRegion:$region);

      let skipDefaultBuilders = 1;
      let builders = [ OpBuilder<
        "ValueRange inputs, ValueRange outputBuffers",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<DotOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange inputs, "
        "ValueRange outputBuffers, ValueRange initTensors",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addOperands(initTensors);
          $_state.addTypes(resultTensorTypes);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<DotOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {}",
        [{
          $_state.addOperands(operands);
          $_state.addAttributes(attributes);
          $_state.addTypes(resultTensorTypes);
          (void)$_state.addRegion();
        }]>
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{ return ::parseNamedStructuredOp<DotOp>(parser, result); }];
      let verifier = [{ return ::verifyNamedStructuredOp(*this); }];
      let hasFolder = 1;
      let hasCanonicalizer = 1;

      let extraClassDeclaration = [{
        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      }];
  }
  def BatchMatmulOp : LinalgStructuredBase_Op<"batch_matmul", [
    NamedStructuredOpTrait,
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"YieldOp">]> {
      let arguments = (ins Variadic<AnyShaped>:$inputs,
                           Variadic<AnyMemRef>:$output_buffers,
                           Variadic<AnyRankedTensor>:$init_tensors);
      let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
      let regions = (region AnyRegion:$region);

      let skipDefaultBuilders = 1;
      let builders = [ OpBuilder<
        "ValueRange inputs, ValueRange outputBuffers",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<BatchMatmulOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange inputs, "
        "ValueRange outputBuffers, ValueRange initTensors",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addOperands(initTensors);
          $_state.addTypes(resultTensorTypes);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<BatchMatmulOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {}",
        [{
          $_state.addOperands(operands);
          $_state.addAttributes(attributes);
          $_state.addTypes(resultTensorTypes);
          (void)$_state.addRegion();
        }]>
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{ return ::parseNamedStructuredOp<BatchMatmulOp>(parser, result); }];
      let verifier = [{ return ::verifyNamedStructuredOp(*this); }];
      let hasFolder = 1;
      let hasCanonicalizer = 1;

      let extraClassDeclaration = [{
        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      }];
  }
  def ConvWOp : LinalgStructuredBase_Op<"conv_1d", [
    NamedStructuredOpTrait,
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"YieldOp">]> {
      let arguments = (ins Variadic<AnyShaped>:$inputs,
                           Variadic<AnyMemRef>:$output_buffers,
                           Variadic<AnyRankedTensor>:$init_tensors);
      let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
      let regions = (region AnyRegion:$region);

      let skipDefaultBuilders = 1;
      let builders = [ OpBuilder<
        "ValueRange inputs, ValueRange outputBuffers",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<ConvWOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange inputs, "
        "ValueRange outputBuffers, ValueRange initTensors",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addOperands(initTensors);
          $_state.addTypes(resultTensorTypes);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<ConvWOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {}",
        [{
          $_state.addOperands(operands);
          $_state.addAttributes(attributes);
          $_state.addTypes(resultTensorTypes);
          (void)$_state.addRegion();
        }]>
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{ return ::parseNamedStructuredOp<ConvWOp>(parser, result); }];
      let verifier = [{ return ::verifyNamedStructuredOp(*this); }];
      let hasFolder = 1;
      let hasCanonicalizer = 1;

      let extraClassDeclaration = [{
        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      }];
  }
  def ConvNWCOp : LinalgStructuredBase_Op<"conv_1d_nwc", [
    NamedStructuredOpTrait,
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"YieldOp">]> {
      let arguments = (ins Variadic<AnyShaped>:$inputs,
                           Variadic<AnyMemRef>:$output_buffers,
                           Variadic<AnyRankedTensor>:$init_tensors);
      let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
      let regions = (region AnyRegion:$region);

      let skipDefaultBuilders = 1;
      let builders = [ OpBuilder<
        "ValueRange inputs, ValueRange outputBuffers",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<ConvNWCOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange inputs, "
        "ValueRange outputBuffers, ValueRange initTensors",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addOperands(initTensors);
          $_state.addTypes(resultTensorTypes);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<ConvNWCOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {}",
        [{
          $_state.addOperands(operands);
          $_state.addAttributes(attributes);
          $_state.addTypes(resultTensorTypes);
          (void)$_state.addRegion();
        }]>
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{ return ::parseNamedStructuredOp<ConvNWCOp>(parser, result); }];
      let verifier = [{ return ::verifyNamedStructuredOp(*this); }];
      let hasFolder = 1;
      let hasCanonicalizer = 1;

      let extraClassDeclaration = [{
        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      }];
  }
  def ConvNCWOp : LinalgStructuredBase_Op<"conv_1d_ncw", [
    NamedStructuredOpTrait,
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"YieldOp">]> {
      let arguments = (ins Variadic<AnyShaped>:$inputs,
                           Variadic<AnyMemRef>:$output_buffers,
                           Variadic<AnyRankedTensor>:$init_tensors);
      let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
      let regions = (region AnyRegion:$region);

      let skipDefaultBuilders = 1;
      let builders = [ OpBuilder<
        "ValueRange inputs, ValueRange outputBuffers",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<ConvNCWOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange inputs, "
        "ValueRange outputBuffers, ValueRange initTensors",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addOperands(initTensors);
          $_state.addTypes(resultTensorTypes);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<ConvNCWOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {}",
        [{
          $_state.addOperands(operands);
          $_state.addAttributes(attributes);
          $_state.addTypes(resultTensorTypes);
          (void)$_state.addRegion();
        }]>
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{ return ::parseNamedStructuredOp<ConvNCWOp>(parser, result); }];
      let verifier = [{ return ::verifyNamedStructuredOp(*this); }];
      let hasFolder = 1;
      let hasCanonicalizer = 1;

      let extraClassDeclaration = [{
        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      }];
  }
  def ConvHWOp : LinalgStructuredBase_Op<"conv_2d", [
    NamedStructuredOpTrait,
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"YieldOp">]> {
      let arguments = (ins Variadic<AnyShaped>:$inputs,
                           Variadic<AnyMemRef>:$output_buffers,
                           Variadic<AnyRankedTensor>:$init_tensors);
      let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
      let regions = (region AnyRegion:$region);

      let skipDefaultBuilders = 1;
      let builders = [ OpBuilder<
        "ValueRange inputs, ValueRange outputBuffers",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<ConvHWOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange inputs, "
        "ValueRange outputBuffers, ValueRange initTensors",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addOperands(initTensors);
          $_state.addTypes(resultTensorTypes);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<ConvHWOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {}",
        [{
          $_state.addOperands(operands);
          $_state.addAttributes(attributes);
          $_state.addTypes(resultTensorTypes);
          (void)$_state.addRegion();
        }]>
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{ return ::parseNamedStructuredOp<ConvHWOp>(parser, result); }];
      let verifier = [{ return ::verifyNamedStructuredOp(*this); }];
      let hasFolder = 1;
      let hasCanonicalizer = 1;

      let extraClassDeclaration = [{
        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      }];
  }
  def ConvNHWCOp : LinalgStructuredBase_Op<"conv_2d_nhwc", [
    NamedStructuredOpTrait,
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"YieldOp">]> {
      let arguments = (ins Variadic<AnyShaped>:$inputs,
                           Variadic<AnyMemRef>:$output_buffers,
                           Variadic<AnyRankedTensor>:$init_tensors);
      let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
      let regions = (region AnyRegion:$region);

      let skipDefaultBuilders = 1;
      let builders = [ OpBuilder<
        "ValueRange inputs, ValueRange outputBuffers",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<ConvNHWCOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange inputs, "
        "ValueRange outputBuffers, ValueRange initTensors",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addOperands(initTensors);
          $_state.addTypes(resultTensorTypes);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<ConvNHWCOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {}",
        [{
          $_state.addOperands(operands);
          $_state.addAttributes(attributes);
          $_state.addTypes(resultTensorTypes);
          (void)$_state.addRegion();
        }]>
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{ return ::parseNamedStructuredOp<ConvNHWCOp>(parser, result); }];
      let verifier = [{ return ::verifyNamedStructuredOp(*this); }];
      let hasFolder = 1;
      let hasCanonicalizer = 1;

      let extraClassDeclaration = [{
        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      }];
  }
  def ConvNCHWOp : LinalgStructuredBase_Op<"conv_2d_nchw", [
    NamedStructuredOpTrait,
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"YieldOp">]> {
      let arguments = (ins Variadic<AnyShaped>:$inputs,
                           Variadic<AnyMemRef>:$output_buffers,
                           Variadic<AnyRankedTensor>:$init_tensors);
      let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
      let regions = (region AnyRegion:$region);

      let skipDefaultBuilders = 1;
      let builders = [ OpBuilder<
        "ValueRange inputs, ValueRange outputBuffers",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<ConvNCHWOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange inputs, "
        "ValueRange outputBuffers, ValueRange initTensors",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addOperands(initTensors);
          $_state.addTypes(resultTensorTypes);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<ConvNCHWOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {}",
        [{
          $_state.addOperands(operands);
          $_state.addAttributes(attributes);
          $_state.addTypes(resultTensorTypes);
          (void)$_state.addRegion();
        }]>
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{ return ::parseNamedStructuredOp<ConvNCHWOp>(parser, result); }];
      let verifier = [{ return ::verifyNamedStructuredOp(*this); }];
      let hasFolder = 1;
      let hasCanonicalizer = 1;

      let extraClassDeclaration = [{
        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      }];
  }
  def ConvDHWOp : LinalgStructuredBase_Op<"conv_3d", [
    NamedStructuredOpTrait,
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"YieldOp">]> {
      let arguments = (ins Variadic<AnyShaped>:$inputs,
                           Variadic<AnyMemRef>:$output_buffers,
                           Variadic<AnyRankedTensor>:$init_tensors);
      let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
      let regions = (region AnyRegion:$region);

      let skipDefaultBuilders = 1;
      let builders = [ OpBuilder<
        "ValueRange inputs, ValueRange outputBuffers",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<ConvDHWOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange inputs, "
        "ValueRange outputBuffers, ValueRange initTensors",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addOperands(initTensors);
          $_state.addTypes(resultTensorTypes);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<ConvDHWOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {}",
        [{
          $_state.addOperands(operands);
          $_state.addAttributes(attributes);
          $_state.addTypes(resultTensorTypes);
          (void)$_state.addRegion();
        }]>
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{ return ::parseNamedStructuredOp<ConvDHWOp>(parser, result); }];
      let verifier = [{ return ::verifyNamedStructuredOp(*this); }];
      let hasFolder = 1;
      let hasCanonicalizer = 1;

      let extraClassDeclaration = [{
        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      }];
  }
  def ConvNDHWCOp : LinalgStructuredBase_Op<"conv_3d_ndhwc", [
    NamedStructuredOpTrait,
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"YieldOp">]> {
      let arguments = (ins Variadic<AnyShaped>:$inputs,
                           Variadic<AnyMemRef>:$output_buffers,
                           Variadic<AnyRankedTensor>:$init_tensors);
      let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
      let regions = (region AnyRegion:$region);

      let skipDefaultBuilders = 1;
      let builders = [ OpBuilder<
        "ValueRange inputs, ValueRange outputBuffers",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<ConvNDHWCOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange inputs, "
        "ValueRange outputBuffers, ValueRange initTensors",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addOperands(initTensors);
          $_state.addTypes(resultTensorTypes);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<ConvNDHWCOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {}",
        [{
          $_state.addOperands(operands);
          $_state.addAttributes(attributes);
          $_state.addTypes(resultTensorTypes);
          (void)$_state.addRegion();
        }]>
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{ return ::parseNamedStructuredOp<ConvNDHWCOp>(parser, result); }];
      let verifier = [{ return ::verifyNamedStructuredOp(*this); }];
      let hasFolder = 1;
      let hasCanonicalizer = 1;

      let extraClassDeclaration = [{
        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      }];
  }
  def ConvNCDHWOp : LinalgStructuredBase_Op<"conv_3d_ncdhw", [
    NamedStructuredOpTrait,
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"YieldOp">]> {
      let arguments = (ins Variadic<AnyShaped>:$inputs,
                           Variadic<AnyMemRef>:$output_buffers,
                           Variadic<AnyRankedTensor>:$init_tensors);
      let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
      let regions = (region AnyRegion:$region);

      let skipDefaultBuilders = 1;
      let builders = [ OpBuilder<
        "ValueRange inputs, ValueRange outputBuffers",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<ConvNCDHWOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange inputs, "
        "ValueRange outputBuffers, ValueRange initTensors",
        [{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addOperands(initTensors);
          $_state.addTypes(resultTensorTypes);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<ConvNCDHWOp>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        }]>, OpBuilder<
        "TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {}",
        [{
          $_state.addOperands(operands);
          $_state.addAttributes(attributes);
          $_state.addTypes(resultTensorTypes);
          (void)$_state.addRegion();
        }]>
      ];
      let printer = [{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{ return ::parseNamedStructuredOp<ConvNCDHWOp>(parser, result); }];
      let verifier = [{ return ::verifyNamedStructuredOp(*this); }];
      let hasFolder = 1;
      let hasCanonicalizer = 1;

      let extraClassDeclaration = [{
        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      }];
  }
