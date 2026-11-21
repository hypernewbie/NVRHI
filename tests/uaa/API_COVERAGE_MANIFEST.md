# NVRHI UAA Public API Coverage Manifest

This manifest records test mappings and open coverage gaps for public interface methods, features, free functions, backend extensions and descriptor fields in the NVRHI public headers (`nvrhi.h`, `common/resource.h`, `utils.h`, `validation.h`, `vulkan.h`, `metal.h`, `d3d11.h`, `d3d12.h`) to its corresponding UAA test ID(s).

## Coverage Status Legend

- **MAPPED**: A current test ID is mapped to this API surface. This status records source-level mapping only; it does not claim the test ran or passed on any backend.
- **GATED_OPTIONAL**: A current test ID is mapped to an optional API surface. The test is intended to gate execution on the advertised feature; this status does not claim a run or pass.
- **GAP_UNIMPLEMENTED**: Public API surface with no test implementation. If no exact registered ID exists for a row, it is a gap even when a related broad test is listed.
- **MISSING_HARDWARE**: Requires optional/preview hardware or SDK support (e.g., Aftermath SDK, NVAPI, Cooperative Vector hardware, DXR 1.2 clusters/OMMs) unavailable on standard test hosts; tested for capability query and safe fallback when absent.

---

## Registration reconciliation (2026-11-20)

The manifest has been reconciled with 191 source registrations. The runner filters by backend and defaults to CPU, so use `--backend metal --list` or `--backend vulkan --list` for GPU cases and `--backend cpu --tag package --list` for package cases. A mapping identifies a relevant assertion in source. Check the run reports for execution status and skips. Rows with no precise case remain gaps.

Known gaps include D3D11/D3D12 native extension methods that have no dedicated cases, enhanced-barrier transitions, direct fast geometry shader and HLSL extension UAV execution, and single-pass stereo execution. The suite does map clearState and internal memory usage in SUB.ListLifecycle, bindless creation in TBL.CreateWrite, and acceleration structure operations in RT.* cases. These mappings still do not claim runtime results across backends.

---

## 1. `nvrhi::IResource` (`include/nvrhi/common/resource.h`)

| Method / Symbol | Test ID(s) | Status | Notes |
|---|---| --- | --- |
| `IResource::AddRef()` | `CPU.RefCounter`, `DEV.CreateDestroy` | MAPPED | Reference increment verified directly and via `RefCountPtr` |
| `IResource::Release()` | `CPU.RefCounter`, `DEV.CreateDestroy` | MAPPED | Reference decrement and zero-count destruction verified |
| `IResource::GetRefCount()` | `CPU.RefCounter`, `NAT.BorrowedOwnership` | MAPPED | Exact refcount invariants checked |
| `IResource::getNativeObject(ObjectType)` | `NAT.Objects`, `NAT.BorrowedOwnership`, `HEAP.Native`, `NAT.VulkanSync`, `NAT.Metal` | MAPPED | Generic objects and Vulkan/Metal extensions are mapped; D3D11/D3D12-specific objects remain gaps. |
| `IResource::queryMemoryRequirements(MemoryRequirements&)` | `HEAP.Requirements`, `BUF.MemoryRequirements`, `DEV.ResourceDescriptions`, `NAT.Metal` | GATED_OPTIONAL | Checked where backend implements resource backing-memory query |

---

## 2. `nvrhi::IDevice` (`include/nvrhi/nvrhi.h`)

| Method | Test ID(s) | Status | Notes |
|---|---| --- | --- |
| `createHeap(const HeapDesc&)` | `HEAP.Create`, `HEAP.Placement`, `HEAP.Reject`, `HEAP.AliasHandoff` | GATED_OPTIONAL | Gated by `Feature::VirtualResources` |
| `queryTopLevelAccelStructPrebuildInfo(...)` | `RT.Prebuild`, `DEV.OptionalFailures` | GATED_OPTIONAL |  |
| `createTexture(const TextureDesc&)` | `TEX.Staging`, `TEX.ClearFloatUInt`, `TEX.UploadReadback`, `TEX.Resolve`, `TEX.DepthStencil`, `STR.Resize` | MAPPED | Covers 1D/2D/3D/Cube/Array/MSAA/UAV/RT/DepthStencil |
| `getTextureMemoryRequirements(ITexture*)` | `HEAP.Requirements`, `HEAP.Subresources`, `SPR.Tiling` | GATED_OPTIONAL | Gated by `Feature::VirtualResources` |
| `bindTextureMemory(ITexture*, IHeap*, uint64_t)` | `HEAP.Placement`, `HEAP.Subresources`, `SPR.Tiling` | GATED_OPTIONAL | Gated by `Feature::VirtualResources` |
| `createHandleForNativeTexture(ObjectType, Object, const TextureDesc&)` | `NAT.ImportBufferTexture`, `TEX.Native`, `NAT.BorrowedOwnership` | MAPPED | Imports native texture handle and verifies non-owning lifetime |
| `createStagingTexture(const TextureDesc&, CpuAccessMode)` | `TEX.Staging`, `TEX.Staging` | MAPPED | Tested for `Read` and `Write` CPU access |
| `mapStagingTexture(IStagingTexture*, const TextureSlice&, CpuAccessMode, size_t*)` | `TEX.Staging`, `TEX.Staging` | MAPPED | Verifies returned row pitch and pixel contents |
| `unmapStagingTexture(IStagingTexture*)` | `TEX.Staging`, `TEX.Staging` | MAPPED | Paired with every `mapStagingTexture` call |
| `getTextureTiling(...)` | `SPR.Tiling` | GATED_OPTIONAL | Tiled resource query where supported |
| `updateTextureTileMappings(...)` | `SPR.Tiling` | GATED_OPTIONAL | Tiled resource mapping where supported |
| `createSamplerFeedbackTexture(ITexture*, const SamplerFeedbackTextureDesc&)` | `SPR.SamplerFeedback` | GATED_OPTIONAL | Gated by `Feature::SamplerFeedback` (D3D12) |
| `createSamplerFeedbackForNativeTexture(ObjectType, Object, ITexture*)` | `SPR.SamplerFeedback`; D3D12 native feedback case is a gap | GATED_OPTIONAL |  |
| `createBuffer(const BufferDesc&)` | `BUF.Create`, `BUF.WriteCopyRead`, `BUF.VolatileConstants`, `BUF.StructuredRaw`, `STR.Repeat` | MAPPED | Covers vertex, index, constant, structured, raw, typed, volatile, virtual |
| `mapBuffer(IBuffer*, CpuAccessMode)` | `BUF.Map`, `BUF.MapInvalid`, `BUF.WriteCopyRead`, `NAT.MappedFlush` | MAPPED | Read and write CPU access paths |
| `unmapBuffer(IBuffer*)` | `BUF.Map`, `BUF.WriteCopyRead`, `NAT.MappedFlush` | MAPPED | Paired with every `mapBuffer` call |
| `getBufferMemoryRequirements(IBuffer*)` | `BUF.MemoryRequirements`, `HEAP.Requirements` | GATED_OPTIONAL | Gated by `Feature::VirtualResources` |
| `bindBufferMemory(IBuffer*, IHeap*, uint64_t)` | `HEAP.Placement`, `HEAP.PendingLifetime` | GATED_OPTIONAL | Gated by `Feature::VirtualResources` |
| `createHandleForNativeBuffer(ObjectType, Object, const BufferDesc&)` | `NAT.ImportBufferTexture`, `NAT.BorrowedOwnership` | MAPPED | Imports native buffer handle and verifies non-owning lifetime |
| `createShader(const ShaderDesc&, const void*, size_t)` | `SHD.Stages`, `SHD.EntryPoints`, `SHD.InvalidBinary`, `SHD.SpirvCompatibility`, `SHD.Shader16Bit`, `CMP.Direct`, `GFX.Draw` | MAPPED | SPIR-V, DXIL, DXBC (where applicable), and Metal entry points |
| `createShaderSpecialization(IShader*, const ShaderSpecialization*, uint32_t)` | `SHD.Specialisation` | GATED_OPTIONAL | Gated by `Feature::ShaderSpecializations` |
| `createShaderLibrary(const void*, size_t)` | `SHD.Library`, `RT.Pipeline` | GATED_OPTIONAL | Tested on backends supporting shader libraries |
| `createSampler(const SamplerDesc&)` | `SMP.Filters`, `SMP.Options` | MAPPED | Linear/point, address modes, border colour, comparison, reduction |
| `createInputLayout(const VertexAttributeDesc*, uint32_t, IShader*)` | `GFX.VertexInput`, `DEV.ResourceDescriptions` | MAPPED | Multiple vertex attributes, strides, offsets, instancing |
| `createEventQuery()` | `QRY.Event`, `QRY.EventQueues`, `QRY.Invalid` | MAPPED | Event query lifecycle and cross-queue signalling |
| `setEventQuery(IEventQuery*, CommandQueue)` | `QRY.Event`, `QRY.EventQueues` | MAPPED | Tested on Graphics/Compute/Copy queues |
| `pollEventQuery(IEventQuery*)` | `QRY.Event`, `QRY.Invalid` | MAPPED | Verified false-to-true completion transition |
| `waitEventQuery(IEventQuery*)` | `QRY.Event`, `QRY.EventQueues`, `SUB.CompletionOrder`, `SUB.MetalHookCompletionError` | MAPPED | Blocking GPU-to-CPU sync, including a simulated Metal completion error |
| `resetEventQuery(IEventQuery*)` | `QRY.Event`, `QRY.Invalid` | MAPPED | Resets query state for reuse |
| `createTimerQuery()` | `QRY.Timer`, `QRY.Unsupported`, `QRY.Invalid` | GATED_OPTIONAL | Timer query creation and lifecycle |
| `pollTimerQuery(ITimerQuery*)` | `QRY.Timer`, `QRY.Unsupported` | GATED_OPTIONAL | Completion polling before/after resolve |
| `getTimerQueryTime(ITimerQuery*)` | `QRY.Timer`, `QRY.Unsupported` | GATED_OPTIONAL | Non-negative elapsed duration check |
| `resetTimerQuery(ITimerQuery*)` | `QRY.Timer`, `QRY.Invalid` | GATED_OPTIONAL | Reuse across dispatches |
| `getGraphicsAPI()` | `DEV.Capabilities`, `NAT.Objects` | MAPPED | Matches active backend |
| `createFramebuffer(const FramebufferDesc&)` | `GFX.Draw`, `GFX.Framebuffer`, `TEX.DepthStencil`, `GFX.Framebuffer` | MAPPED | Single/MRT colour, depth-stencil, mip/slice attachments |
| `createGraphicsPipeline(const GraphicsPipelineDesc&, const FramebufferInfo&)` | `GFX.Draw`, `GFX.Blend`, `GFX.Framebuffer`, `TBL.GraphicsMultipleLayouts`, `STR.Soak` | MAPPED | Primary `FramebufferInfo` overload |
| `createGraphicsPipeline(const GraphicsPipelineDesc&, IFramebuffer*)` | `GFX.Framebuffer` | MAPPED | Deprecated `IFramebuffer*` overload compatibility |
| `createComputePipeline(const ComputePipelineDesc&)` | `CMP.Direct`, `SHD.PipelineFailure`, `TBL.MultipleLayouts`, `TBL.MultipleLayoutsExplicit`, `STR.Repeat`, `BND.BindingSet` | MAPPED | Compute PSO creation and dispatch |
| `createMeshletPipeline(const MeshletPipelineDesc&, const FramebufferInfo&)` | `MSH.Direct`, `MSH.TaskPayload`, `MSH.Indirect` | GATED_OPTIONAL | Gated by `Feature::Meshlets` |
| `createMeshletPipeline(const MeshletPipelineDesc&, IFramebuffer*)` | `GFX.Framebuffer`, `MSH.Direct` | GATED_OPTIONAL | Deprecated `IFramebuffer*` overload, gated by `Feature::Meshlets` |
| `createRayTracingPipeline(const rt::PipelineDesc&)` | `RT.Pipeline`, `RT.ShaderTable` | GATED_OPTIONAL | Gated by `Feature::RayTracingPipeline` |
| `createBindingLayout(const BindingLayoutDesc&)` | `BND.Layout`, `BND.Ranges`, `BND.StageIsolation`, `BND.PushConstants` | MAPPED | Regular binding layouts across spaces/sets |
| `createBindlessLayout(const BindlessLayoutDesc&)` | `TBL.CreateWrite`, `TBL.UpdateAfterBind`, `TBL.MetalUabRejected`, `STR.Resize` | GATED_OPTIONAL | Metal rejects update-after-bind layouts. Other layout paths run where supported. |
| `createBindingSet(const BindingSetDesc&, IBindingLayout*)` | `BND.BindingSet`, `BND.BindingLifetime`, `BND.Invalid`, `UTL.BindingSet` | MAPPED | All resource binding types + `trackLiveness` |
| `createDescriptorTable(IBindingLayout*)` | `TBL.CreateWrite`, `TBL.CreateWrite`, `STR.Resize` | GATED_OPTIONAL | Gated on bindless-capable backends |
| `resizeDescriptorTable(IDescriptorTable*, uint32_t, bool)` | `TBL.CreateWrite`, `STR.Resize` | GATED_OPTIONAL | Grow, shrink, `keepContents` true/false |
| `writeDescriptorTable(IDescriptorTable*, const BindingSetItem&)` | `TBL.CreateWrite`, `TBL.FixedArray`, `TBL.LargeOrdinary`, `TBL.Samplers`, `TBL.TypedUav`, `TBL.UpdateAfterBind`, `TBL.PreSubmitUpdate`, `TBL.UintSparse`, `STR.Resize` | GATED_OPTIONAL | Metal does not support update-after-bind. The validation layer rejects bindless samplers. |
| `createOpacityMicromap(const rt::OpacityMicromapDesc&)` | `RT.OpacityMicromap`, `DEV.OptionalFailures` | GATED_OPTIONAL | Gated by `Feature::RayTracingOpacityMicromap` |
| `createAccelStruct(const rt::AccelStructDesc&)` | `RT.BottomLevel`, `RT.TopLevel`, `RT.InlineQuery`, `RT.Pipeline`, `STA.RayStructures` | GATED_OPTIONAL |  |
| `getAccelStructMemoryRequirements(rt::IAccelStruct*)` | `RT.Placement`, `HEAP.Requirements` | GATED_OPTIONAL |  |
| `getClusterOperationSizeInfo(const rt::cluster::OperationParams&)` | `RT.Clusters`, `DEV.OptionalFailures` | GATED_OPTIONAL | Gated by `Feature::RayTracingClusters` |
| `bindAccelStructMemory(rt::IAccelStruct*, IHeap*, uint64_t)` | `RT.Placement`, `HEAP.Placement` | GATED_OPTIONAL |  |
| `createCommandList(const CommandListParameters&)` | `DEV.CreateDestroy`, `SUB.QueueDependencies`, `STR.Threads` | MAPPED | Graphics, Compute, Copy, immediate vs deferred |
| `executeCommandLists(ICommandList* const*, size_t, CommandQueue)` | `SUB.ConcurrentSubmission`, `SUB.CompletionOrder`, `SUB.MetalMissingCommandBuffer`, `SUB.MetalHookSerialOrder`, `STR.InFlight` | MAPPED | Batch, concurrent and multi-queue submission; the Metal hook forces the serial-to-commit interval. |
| `queueWaitForCommandList(CommandQueue, CommandQueue, uint64_t)` | `SUB.QueueDependencies`, `SUB.CompletionOrder`, `SUB.MetalHookWaitAllocation`, `SUB.MetalHookCompletionError` | GATED_OPTIONAL | Metal tests include allocation and completion failure injection. |
| `waitForIdle()` | `DEV.CreateDestroy`, `SUB.MetalHookCompletionError`, `STR.InFlight`, `STR.Resize`, `STR.Threads` | MAPPED | Full queue drain and simulated Metal completion failure |
| `createCommandListLifetimeTracker(CommandQueue)` | `SUB.LifetimeTracker`, `STR.Threads` | GATED_OPTIONAL | Supported on Vulkan and D3D12; null/error on Metal and D3D11 |
| `runGarbageCollection()` | `DEV.GarbageCollection`, `BUF.Lifetime`, `STR.Resize`, `STR.Soak` | MAPPED | Deferred resource reclamation |
| `queryFeatureSupport(Feature, void*, size_t)` | `DEV.Capabilities`, `OPT.VRS`, `OPT.Wave`, `RT.Clusters` | MAPPED | Queries all 27 features + info structs |
| `queryFormatSupport(Format)` | `DEV.FormatSupport`, `UTL.ChooseFormat` | MAPPED | Checks `FormatSupport` flags across formats |
| `queryCoopVecFeatures()` | `COOP.Capabilities` | GATED_OPTIONAL | Deprecated aggregate query validated when CoopVec present |
| `queryCoopVecMatMulFormatSupport(const coopvec::MatMulFormatCombo&)` | `COOP.Capabilities` | GATED_OPTIONAL | Gated by `CooperativeVectorInferencing` / `Training` |
| `queryCoopVecTrainingFormatSupport(coopvec::DataType)` | `COOP.Capabilities` | GATED_OPTIONAL | Gated by `CooperativeVectorTraining` |
| `getCoopVecMatrixSize(coopvec::DataType, coopvec::MatrixLayout, int, int)` | `COOP.Capabilities`, `COOP.Convert` | GATED_OPTIONAL | Matrix byte size calculation |
| `getNativeQueue(ObjectType, CommandQueue)` | `NAT.Objects`, `NAT.VulkanSync`, `NAT.Metal`; D3D12 native queue case is a gap | MAPPED |  |
| `getMessageCallback()` | `DEV.CreateDestroy`, `VAL.ErrorCapture` | MAPPED | Verifies registered `IMessageCallback*` |
| `isAftermathEnabled()` | `NAT.Aftermath` | MAPPED | Checks Aftermath status (typically false unless SDK enabled) |
| `getAftermathCrashDumpHelper()` | `NAT.Aftermath` | MISSING_HARDWARE | Tested via query/reference when Aftermath is enabled |
| `executeCommandList(ICommandList*, CommandQueue)` | `CMP.Direct`, `GFX.Draw`, `STR.InFlight` | MAPPED | Inline single-list wrapper around `executeCommandLists` |

---

## 3. `nvrhi::ICommandList` (`include/nvrhi/nvrhi.h`)

| Method | Test ID(s) | Status | Notes |
|---|---| --- | --- |
| `open()` | `DEV.CreateDestroy`, `VAL.CommandOrder`, `STR.Repeat` | MAPPED | Command recording open/re-open |
| `close()` | `DEV.CreateDestroy`, `VAL.CommandOrder`, `STR.Repeat` | MAPPED | Command recording finalise |
| `clearState()` | `SUB.ListLifecycle` | MAPPED | Lifecycle case clears command state and continues recording. |
| `clearTextureFloat(ITexture*, TextureSubresourceSet, const Color&)` | `TEX.ClearFloatUInt`, `UTL.ClearAttachments` | MAPPED | Colour RT/UAV float clear |
| `clearDepthStencilTexture(ITexture*, TextureSubresourceSet, bool, float, bool, uint8_t)` | `TEX.DepthStencil`, `UTL.ClearAttachments` | MAPPED | Depth and stencil clears |
| `clearTextureUInt(ITexture*, TextureSubresourceSet, uint32_t)` | `TEX.ClearFloatUInt` | MAPPED | Integer UAV/RT texture clear |
| `copyTexture(ITexture*, const TextureSlice&, ITexture*, const TextureSlice&)` | `TEX.Staging`, `TEX.UploadReadback` | MAPPED | GPU-to-GPU 2D/mip/slice texture copy |
| `copyTexture(IStagingTexture*, const TextureSlice&, ITexture*, const TextureSlice&)` | `TEX.Staging`, `TEX.Staging` | MAPPED | GPU texture to readback staging texture |
| `copyTexture(ITexture*, const TextureSlice&, IStagingTexture*, const TextureSlice&)` | `TEX.Staging`, `TEX.Staging` | MAPPED | Upload staging texture to GPU texture |
| `writeTexture(ITexture*, uint32_t, uint32_t, const void*, size_t, size_t)` | `TEX.Staging`, `TEX.UploadReadback`, `TEX.Views` | MAPPED | Direct CPU-to-texture upload |
| `resolveTexture(ITexture*, const TextureSubresourceSet&, ITexture*, const TextureSubresourceSet&)` | `TEX.Resolve` | MAPPED | MSAA-to-non-MSAA resolve |
| `writeBuffer(IBuffer*, const void*, size_t, uint64_t)` | `BUF.WriteCopyRead`, `BUF.Overwrite`, `BUF.VolatileConstants`, `STR.Repeat` | MAPPED | Static, volatile, and offset buffer writes |
| `clearBufferUInt(IBuffer*, uint32_t)` | `BUF.Clear` | MAPPED | Full buffer uint32 clear |
| `copyBuffer(IBuffer*, uint64_t, IBuffer*, uint64_t, uint64_t)` | `BUF.WriteCopyRead`, `BUF.Clear` | MAPPED | Buffer-to-buffer copy with offsets |
| `clearSamplerFeedbackTexture(ISamplerFeedbackTexture*)` | `SPR.SamplerFeedback` | GATED_OPTIONAL | Gated by `Feature::SamplerFeedback` (D3D12) |
| `decodeSamplerFeedbackTexture(IBuffer*, ISamplerFeedbackTexture*, Format)` | `SPR.SamplerFeedback` | GATED_OPTIONAL | Gated by `Feature::SamplerFeedback` (D3D12) |
| `setSamplerFeedbackTextureState(ISamplerFeedbackTexture*, ResourceStates)` | `SPR.SamplerFeedback` | GATED_OPTIONAL | Gated by `Feature::SamplerFeedback` (D3D12) |
| `setPushConstants(const void*, size_t)` | `BND.PushConstants`, `GFX.Draw`, `STR.Soak` | MAPPED | Compute and graphics push constants |
| `setGraphicsState(const GraphicsState&)` | `GFX.Draw`, `GFX.VertexInput`, `STR.Soak` | MAPPED | Multiple graphics cases set and exercise graphics state. |
| `draw(const DrawArguments&)` | `GFX.Draw`, `GFX.Blend`, `STR.Soak` | MAPPED | Non-indexed and instanced draw |
| `drawIndexed(const DrawArguments&)` | `GFX.DrawArguments`, `GFX.VertexInput`, `BUF.VertexIndexIndirect` | MAPPED | Indexed draw with offsets |
| `drawIndirect(uint32_t, uint32_t)` | `GFX.Indirect`, `BUF.VertexIndexIndirect` | MAPPED | Single and multi-draw indirect |
| `drawIndexedIndirect(uint32_t, uint32_t)` | `GFX.Indirect` | MAPPED | Indexed indirect draw |
| `drawIndexedIndirectCount(uint32_t, uint32_t, uint32_t)` | `GFX.IndirectCount` | GATED_OPTIONAL | Indirect count draw where supported |
| `setComputeState(const ComputeState&)` | `CMP.Direct`, `CMP.StateChanges`, `STR.Repeat`, `STR.Threads` | MAPPED | Compute pipeline, bindings, indirect buffer |
| `dispatch(uint32_t, uint32_t, uint32_t)` | `CMP.Direct`, `CMP.Dependencies`, `STR.Repeat`, `STR.InFlight` | MAPPED | 1D/2D/3D compute dispatches |
| `dispatchIndirect(uint32_t)` | `CMP.Indirect`, `CMP.IndirectValidation` | MAPPED | GPU-generated `DispatchIndirectArguments` |
| `setMeshletState(const MeshletState&)` | `MSH.Direct`, `MSH.TaskPayload`, `MSH.Indirect` | GATED_OPTIONAL | Gated by `Feature::Meshlets` |
| `dispatchMesh(uint32_t, uint32_t, uint32_t)` | `MSH.Direct`, `MSH.TaskPayload` | GATED_OPTIONAL | Gated by `Feature::Meshlets` |
| `dispatchMeshIndirect(uint32_t, uint32_t)` | `MSH.Indirect` | GATED_OPTIONAL | Gated by `Feature::Meshlets` |
| `dispatchMeshIndirectCount(uint32_t, uint32_t, uint32_t)` | `MSH.Indirect` | GATED_OPTIONAL | Gated by `Feature::Meshlets` & backend count support (Vulkan) |
| `setRayTracingState(const rt::State&)` | `RT.Pipeline`, `RT.ShaderTable` | GATED_OPTIONAL | Gated by `Feature::RayTracingPipeline` |
| `dispatchRays(const rt::DispatchRaysArguments&)` | `RT.Pipeline`, `RT.ShaderTable` | GATED_OPTIONAL | Gated by `Feature::RayTracingPipeline` |
| `buildOpacityMicromap(rt::IOpacityMicromap*, const rt::OpacityMicromapDesc&)` | `RT.OpacityMicromap` | GATED_OPTIONAL | Gated by `Feature::RayTracingOpacityMicromap` |
| `buildBottomLevelAccelStruct(...)` | `RT.BottomLevel`, `RT.TopLevel`, `RT.InlineQuery`, `RT.Pipeline`, `UTL.RayBuild` | GATED_OPTIONAL |  |
| `compactBottomLevelAccelStructs()` | `RT.BottomLevel`, `RT.CopyCompact` | GATED_OPTIONAL |  |
| `copyRaytracingAccelerationStructure(rt::IAccelStruct*, rt::IAccelStruct*)` | `RT.BottomLevel`, `RT.CopyCompact` | GATED_OPTIONAL |  |
| `buildTopLevelAccelStruct(...)` | `RT.BottomLevel`, `RT.TopLevel`, `RT.InlineQuery`, `RT.Pipeline` | GATED_OPTIONAL |  |
| `executeMultiIndirectClusterOperation(const rt::cluster::OperationDesc&)` | `RT.Clusters` | GATED_OPTIONAL | Gated by `Feature::RayTracingClusters` |
| `buildTopLevelAccelStructFromBuffer(...)` | `RT.FromBuffer` | GATED_OPTIONAL | Gated by `Feature::RayTracingAccelStruct` |
| `convertCoopVecMatrices(const coopvec::ConvertMatrixLayoutDesc*, size_t)` | `COOP.Convert` | GATED_OPTIONAL | Gated by `CooperativeVectorInferencing` / `Training` |
| `beginTimerQuery(ITimerQuery*)` | `QRY.Timer`, `QRY.Invalid` | GATED_OPTIONAL | Paired with `endTimerQuery` |
| `endTimerQuery(ITimerQuery*)` | `QRY.Timer`, `QRY.Unsupported` | GATED_OPTIONAL | Resolves timestamp query |
| `beginMarker(const char*)` | `QRY.Markers` | MAPPED | Nested debug marker regions |
| `endMarker()` | `QRY.Markers` | MAPPED | Paired with `beginMarker` |
| `setEnableAutomaticBarriers(bool)` | `STA.Disabled`, `STA.Initial`, `STA.Commit` | MAPPED | Automatic vs manual barrier mode |
| `setResourceStatesForBindingSet(IBindingSet*)` | `STA.BindingTransitions`, `STA.Commit` | MAPPED | Explicit binding-set state transitions |
| `setResourceStatesForFramebuffer(IFramebuffer*)` | `STA.BindingTransitions`, `GFX.Draw` | MAPPED | Non-virtual `NVRHI_API` helper on `ICommandList` |
| `setEnableUavBarriersForTexture(ITexture*, bool)` | `STA.Uav`, `UTL.Barriers` | MAPPED | UAV barrier toggling for textures |
| `setEnableUavBarriersForBuffer(IBuffer*, bool)` | `STA.Uav`, `UTL.Barriers` | MAPPED | UAV barrier toggling for buffers |
| `beginTrackingTextureState(ITexture*, TextureSubresourceSet, ResourceStates)` | `STA.Initial`, `STA.PerSubresource` | MAPPED | Explicit texture state tracking |
| `beginTrackingBufferState(IBuffer*, ResourceStates)` | `STA.Initial`, `STA.Disabled` | MAPPED | Explicit buffer state tracking |
| `setTextureState(ITexture*, TextureSubresourceSet, ResourceStates)` | `STA.PerSubresource`, `STA.Commit` | MAPPED | Explicit texture barrier placement |
| `setBufferState(IBuffer*, ResourceStates)` | `STA.Commit`, `STA.Disabled` | MAPPED | Explicit buffer barrier placement |
| `setAccelStructState(rt::IAccelStruct*, ResourceStates)` | `STA.RayStructures`, `RT.BottomLevel`, `RT.CopyCompact` | GATED_OPTIONAL |  |
| `setPermanentTextureState(ITexture*, ResourceStates)` | `STA.Permanent` | MAPPED | Permanent texture state + validation check |
| `setPermanentBufferState(IBuffer*, ResourceStates)` | `STA.Permanent` | MAPPED | Permanent buffer state + validation check |
| `commitBarriers()` | `STA.Commit`, `STA.Permanent` | MAPPED | Explicit barrier flush |
| `getTextureSubresourceState(ITexture*, ArraySlice, MipLevel)` | `STA.Initial`, `STA.PerSubresource` | MAPPED | Tracked subresource state inspection |
| `getBufferState(IBuffer*)` | `STA.Initial`, `STA.Permanent` | MAPPED | Tracked buffer state inspection |
| `getDevice()` | `DEV.CreateDestroy`, `SUB.QueueDependencies` | MAPPED | Non-AddRef owning device pointer |
| `getDesc()` | `DEV.ResourceDescriptions`, `SUB.QueueDependencies` | MAPPED | Returns `CommandListParameters` |
| `getInternalMemoryUsage()` | `SUB.ListLifecycle` | MAPPED | The lifecycle test checks upload and scratch memory counters. |

---

## 4. Resource & Pipeline Interfaces (`include/nvrhi/nvrhi.h`)

| Interface & Method | Test ID(s) | Status | Notes |
|---|---| --- | --- |
| `IHeap::getDesc()` | `HEAP.Create`, `DEV.ResourceDescriptions` | GATED_OPTIONAL | Gated by `Feature::VirtualResources` |
| `ITexture::getDesc()` | `DEV.ResourceDescriptions`, `TEX.Staging` | MAPPED | Dimension, format, mip, sample count round-trip |
| `ITexture::getNativeView(...)` | `NAT.NativeViews`, `NAT.Objects` | MAPPED | Subresource & component mapping override native view |
| `IStagingTexture::getDesc()` | `DEV.ResourceDescriptions`, `TEX.Staging` | MAPPED | Staging texture descriptor inspection |
| `ISamplerFeedbackTexture::getDesc()` | `SPR.SamplerFeedback` | GATED_OPTIONAL | Gated by `Feature::SamplerFeedback` |
| `ISamplerFeedbackTexture::getPairedTexture()` | `SPR.SamplerFeedback` | GATED_OPTIONAL | Returns paired texture handle |
| `IInputLayout::getNumAttributes()` | `DEV.ResourceDescriptions`, `GFX.VertexInput` | MAPPED | Matches attribute array count |
| `IInputLayout::getAttributeDesc(uint32_t)` | `DEV.ResourceDescriptions`, `GFX.VertexInput` | MAPPED | Validates attribute name, format, stride, offset |
| `IBuffer::getDesc()` | `DEV.ResourceDescriptions`, `BUF.Create` | MAPPED | ByteSize, stride, flags round-trip |
| `IBuffer::getGpuVirtualAddress()` | `RT.AdvancedGeometry`, `RT.FromBuffer` | GATED_OPTIONAL | Native address use is exercised for supported ray tracing paths. |
| `IShader::getDesc()` | `DEV.ResourceDescriptions`, `SHD.Stages` | MAPPED | Type, entry name, debug name round-trip |
| `IShader::getBytecode(const void**, size_t*)` | `SHD.Stages`, `SHD.Library` | MAPPED | Non-null bytecode pointer and size |
| `IShaderLibrary::getBytecode(const void**, size_t*)` | `SHD.Library` | GATED_OPTIONAL | Tested on shader-library-capable backends |
| `IShaderLibrary::getShader(const char*, ShaderType)` | `SHD.Library`, `RT.Pipeline` | GATED_OPTIONAL | Extracts named entry points from library |
| `ISampler::getDesc()` | `DEV.ResourceDescriptions`, `SMP.Filters` | MAPPED | Filter, address, reduction, compareFunc round-trip |
| `IFramebuffer::getDesc()` | `DEV.ResourceDescriptions`, `GFX.Framebuffer` | MAPPED | Colour/depth attachment inspection |
| `IFramebuffer::getFramebufferInfo()` | `GFX.Framebuffer`, `GFX.Framebuffer` | MAPPED | Width, height, formats, sample count |
| `IBindingLayout::getDesc()` | `DEV.ResourceDescriptions`, `BND.Layout` | MAPPED | Non-null for regular, null for bindless |
| `IBindingLayout::getBindlessDesc()` | `DEV.ResourceDescriptions`, `TBL.CreateWrite` | GATED_OPTIONAL | Non-null for bindless, null for regular |
| `IBindingSet::getDesc()` | `DEV.ResourceDescriptions`, `BND.BindingSet` | MAPPED | Non-null for binding sets, null for descriptor tables |
| `IBindingSet::getLayout()` | `DEV.ResourceDescriptions`, `BND.BindingSet` | MAPPED | Returns parent `IBindingLayout*` |
| `IDescriptorTable::getCapacity()` | `TBL.CreateWrite`, `STR.Resize` | GATED_OPTIONAL | Current table capacity check |
| `IDescriptorTable::getFirstDescriptorIndexInHeap()` | `TBL.CreateWrite`; D3D12 heap-index case is a gap | GATED_OPTIONAL |  |
| `IGraphicsPipeline::getDesc()` | `DEV.ResourceDescriptions`, `GFX.Draw` | MAPPED | Graphics pipeline descriptor round-trip |
| `IGraphicsPipeline::getFramebufferInfo()` | `GFX.Framebuffer`, `GFX.Draw` | MAPPED | Compatible `FramebufferInfo` check |
| `IComputePipeline::getDesc()` | `DEV.ResourceDescriptions`, `CMP.Direct` | MAPPED | Compute pipeline descriptor round-trip |
| `IMeshletPipeline::getDesc()` | `MSH.Direct`, `DEV.ResourceDescriptions` | GATED_OPTIONAL | Gated by `Feature::Meshlets` |
| `IMeshletPipeline::getFramebufferInfo()` | `MSH.Direct`, `GFX.Framebuffer` | GATED_OPTIONAL | Gated by `Feature::Meshlets` |
| `ICommandListLifetimeTracker::runGarbageCollection()` | `SUB.LifetimeTracker`, `STR.Threads` | GATED_OPTIONAL | Supported on Vulkan and D3D12 |
| `rt::IOpacityMicromap::getDesc()` | `RT.OpacityMicromap` | GATED_OPTIONAL | Gated by `Feature::RayTracingOpacityMicromap` |
| `rt::IOpacityMicromap::isCompacted()` | `RT.OpacityMicromap` | GATED_OPTIONAL | Gated by `Feature::RayTracingOpacityMicromap` |
| `rt::IOpacityMicromap::getDeviceAddress()` | `RT.OpacityMicromap` | GATED_OPTIONAL | Gated by `Feature::RayTracingOpacityMicromap` |
| `rt::IAccelStruct::getDesc()` | `RT.BottomLevel`, `RT.CopyCompact` | GATED_OPTIONAL |  |
| `rt::IAccelStruct::isCompacted()` | `RT.BottomLevel`, `RT.CopyCompact` | GATED_OPTIONAL |  |
| `rt::IAccelStruct::getDeviceAddress()` | `RT.FromBuffer` | GATED_OPTIONAL |  |
| `rt::IShaderTable::getDesc()` | `RT.ShaderTable` | GATED_OPTIONAL | Cached vs uncached shader table desc |
| `rt::IShaderTable::getNumEntries()` | `RT.ShaderTable` | GATED_OPTIONAL | Entry count after add/clear |
| `rt::IShaderTable::getPipeline()` | `RT.ShaderTable` | GATED_OPTIONAL | Owning ray tracing pipeline |
| `rt::IShaderTable::setRayGenerationShader(...)` | `RT.Pipeline`, `RT.ShaderTable` | GATED_OPTIONAL | Sets raygen shader + local bindings |
| `rt::IShaderTable::addMissShader(...)` | `RT.Pipeline`, `RT.ShaderTable` | GATED_OPTIONAL | Appends miss shader |
| `rt::IShaderTable::addHitGroup(...)` | `RT.Pipeline`, `RT.ShaderTable` | GATED_OPTIONAL | Appends hit group |
| `rt::IShaderTable::addCallableShader(...)` | `RT.Pipeline`, `RT.ShaderTable` | GATED_OPTIONAL | Appends callable shader |
| `rt::IShaderTable::clearMissShaders()` | `RT.ShaderTable` | GATED_OPTIONAL | Clears miss entries and updates `getNumEntries()` |
| `rt::IShaderTable::clearHitShaders()` | `RT.ShaderTable` | GATED_OPTIONAL | Clears hit group entries |
| `rt::IShaderTable::clearCallableShaders()` | `RT.ShaderTable` | GATED_OPTIONAL | Clears callable entries |
| `rt::IPipeline::getDesc()` | `RT.Pipeline` | GATED_OPTIONAL | Ray tracing pipeline desc round-trip |
| `rt::IPipeline::createShaderTable(...)` | `RT.Pipeline`, `RT.ShaderTable` | GATED_OPTIONAL | Uncached and cached shader tables |
| `IEventQuery` (interface) | `QRY.Event`, `QRY.EventQueues` | MAPPED | Inherited `IResource` lifecycle |
| `ITimerQuery` (interface) | `QRY.Timer`, `QRY.Unsupported` | GATED_OPTIONAL | Inherited `IResource` lifecycle |

---

## 5. `nvrhi::Feature` Enum Values (All 27)

| Enum Value | Test ID(s) | Status | Notes |
|---|---| --- | --- |
| `Feature::ComputeQueue` | `DEV.Capabilities`, `CMP.MultiQueue`, `SUB.QueueDependencies`, `QRY.EventQueues` | MAPPED | Queried on all; multi-queue tested when true |
| `Feature::ConservativeRasterization` | `DEV.Capabilities`, `OPT.Raster` | GATED_OPTIONAL | Raster test when supported; clean skip/check otherwise |
| `Feature::ConstantBufferRanges` | `DEV.Capabilities`, `BND.Ranges` | MAPPED | Partial CB offset/size binding verified when true |
| `Feature::CopyQueue` | `DEV.Capabilities`, `SUB.QueueDependencies`, `QRY.EventQueues` | MAPPED | Queried on all; copy queue tested when true |
| `Feature::DeferredCommandLists` | `DEV.Capabilities`, `STR.Threads` | MAPPED | Multithreaded recording on Vulkan/D3D12; skipped cleanly on Metal/D3D11 |
| `Feature::FastGeometryShader` | `DEV.Capabilities` | GATED_OPTIONAL |  |
| `Feature::HeapDirectlyIndexed` | `DEV.Capabilities` | GATED_OPTIONAL |  |
| `Feature::HlslExtensionUAV` | `DEV.Capabilities` | GATED_OPTIONAL |  |
| `Feature::LinearSweptSpheres` | `DEV.Capabilities`, `RT.AdvancedGeometry` | GATED_OPTIONAL | LSS BLAS geometry when supported |
| `Feature::Meshlets` | `DEV.Capabilities`, `MSH.Direct`, `MSH.TaskPayload`, `MSH.Indirect` | GATED_OPTIONAL | Amplification + Mesh pipeline execution |
| `Feature::RayQuery` | `DEV.Capabilities`, `RT.InlineQuery` | GATED_OPTIONAL | Inline compute `TraceRayInline` when supported |
| `Feature::RayTracingAccelStruct` | `DEV.Capabilities`, `RT.BottomLevel`, `RT.TopLevel`, `RT.FromBuffer` | GATED_OPTIONAL | Capability query and gated acceleration structure cases. |
| `Feature::RayTracingClusters` | `DEV.Capabilities`, `RT.Clusters` | GATED_OPTIONAL | `RayTracingClustersFeatureInfo` + cluster ops when supported |
| `Feature::RayTracingOpacityMicromap` | `DEV.Capabilities`, `RT.OpacityMicromap` | GATED_OPTIONAL | OMM creation/build when supported |
| `Feature::RayTracingPipeline` | `DEV.Capabilities`, `RT.Pipeline`, `RT.ShaderTable` | GATED_OPTIONAL | RayGen/Miss/ClosestHit/AnyHit dispatch |
| `Feature::RayTracingPositionFetch` | `DEV.Capabilities`, `RT.Pipeline` | GATED_OPTIONAL | Position fetch query and optional raygen check |
| `Feature::SamplerFeedback` | `DEV.Capabilities`, `SPR.SamplerFeedback` | GATED_OPTIONAL | D3D12 sampler feedback texture + decode |
| `Feature::ShaderExecutionReordering` | `DEV.Capabilities`, `RT.Reordering` | GATED_OPTIONAL | SER capability query |
| `Feature::ShaderSpecializations` | `DEV.Capabilities`, `SHD.Specialisation` | GATED_OPTIONAL | SPIR-V specialisation constants |
| `Feature::SinglePassStereo` | `DEV.Capabilities` | GATED_OPTIONAL |  |
| `Feature::Spheres` | `DEV.Capabilities`, `RT.AdvancedGeometry` | GATED_OPTIONAL | Sphere BLAS geometry when supported |
| `Feature::VariableRateShading` | `DEV.Capabilities`, `OPT.VRS` | GATED_OPTIONAL | `VariableRateShadingFeatureInfo` + VRS draw |
| `Feature::VirtualResources` | `DEV.Capabilities`, `HEAP.Create`, `HEAP.Placement`, `HEAP.Requirements`, `SPR.Tiling` | GATED_OPTIONAL | Virtual buffers/textures + explicit heap binding |
| `Feature::WaveLaneCountMinMax` | `DEV.Capabilities`, `OPT.Wave` | GATED_OPTIONAL | `WaveLaneCountMinMaxFeatureInfo` + `cs_wave_lane` |
| `Feature::CooperativeVectorInferencing` | `DEV.Capabilities`, `COOP.Capabilities`, `COOP.Convert` | MISSING_HARDWARE | Queried and validated when hardware supports CoopVec |
| `Feature::CooperativeVectorTraining` | `DEV.Capabilities`, `COOP.Capabilities` | MISSING_HARDWARE | Training format support queried |
| `Feature::EnhancedBarriers` | `DEV.Capabilities` | GATED_OPTIONAL |  |

---

## 6. Public Free Functions & Utility Classes

### `include/nvrhi/nvrhi.h`

| Function / Helper | Test ID(s) | Status | Notes |
|---|---| --- | --- |
| `nvrhi::verifyHeaderVersion(uint32_t)` | `CPU.ObjectAndEnums`, `PKG.Build` | MAPPED | Tested with `c_HeaderVersion`, mismatching value, and consumer link |
| `nvrhi::getFormatInfo(Format)` | `CPU.FormatMetadata`, `CPU.FormatBoundaries` | MAPPED | All `Format` entries up to `Format::COUNT` |
| `nvrhi::packComponentMapping(std::optional<ComponentMapping>)` | `CPU.DescriptorDefaults`, `TEX.Views` | MAPPED | `std::nullopt` vs explicit `ComponentMapping` |
| `nvrhi::resolveComponentMapping(uint16_t, const ComponentMapping&)` | `CPU.DescriptorDefaults`, `TEX.Views` | MAPPED | Packed override vs texture default fallback |
| `nvrhi::resolveComponentMapping(std::optional<ComponentMapping>, const ComponentMapping&)` | `CPU.DescriptorDefaults` | MAPPED | Optional override vs texture default fallback |
| `nvrhi::hash_combine(size_t&, const T&)` | `CPU.Containers` | MAPPED | Verified with `std::hash` specialisations (`BindingSetDesc`, `FramebufferInfo`, etc.) |
| `nvrhi::coopvec::getDataTypeSize(DataType)` | `CPU.ObjectAndEnums`, `COOP.Capabilities` | MAPPED | Byte widths across all `coopvec::DataType` entries |
| `nvrhi::coopvec::getOptimalMatrixStride(...)` | `CPU.ObjectAndEnums`, `COOP.Capabilities` | MAPPED | RowMajor/ColumnMajor/Optimal strides |
| `TextureSlice::resolve(const TextureDesc&) const` | `CPU.Slices` | MAPPED | `-1` sentinel resolution and mip scaling |
| `TextureSubresourceSet::resolve(const TextureDesc&, bool) const` | `CPU.Subresources` | MAPPED | `AllMipLevels` / `AllArraySlices` resolution |
| `TextureSubresourceSet::isEntireTexture(const TextureDesc&) const` | `CPU.Subresources` | MAPPED | Full vs partial subresource checks |
| `BufferRange::resolve(const BufferDesc&) const` | `CPU.BufferRanges` | MAPPED | `EntireBuffer` and partial range resolution |
| `BlendState::RenderTarget::usesConstantColor() const` | `CPU.DescriptorDefaults`, `GFX.Blend` | MAPPED | Constant blend factor detection |
| `FramebufferInfo::FramebufferInfo(const FramebufferDesc&)` | `CPU.DescriptorDefaults`, `GFX.Framebuffer` | MAPPED | Derived from framebuffer attachments |
| `FramebufferInfoEx::FramebufferInfoEx(const FramebufferDesc&)` | `CPU.DescriptorDefaults`, `GFX.Framebuffer` | MAPPED | Extended dimensions & `getViewport()` |

### `include/nvrhi/utils.h`

| Function / Class | Test ID(s) | Status | Notes |
|---|---| --- | --- |
| `utils::CreateAddBlendState(BlendFactor, BlendFactor)` | `UTL.Descriptors`, `CPU.DescriptorDefaults` | MAPPED | Additive blend state builder |
| `utils::CreateStaticConstantBufferDesc(uint32_t, const char*)` | `UTL.Descriptors`, `STR.Repeat` | MAPPED | Static CB descriptor defaults |
| `utils::CreateVolatileConstantBufferDesc(uint32_t, const char*, uint32_t)` | `UTL.Descriptors`, `BUF.VolatileConstants` | MAPPED | Volatile CB descriptor defaults |
| `utils::CreateBindingSetAndLayout(...)` | `UTL.BindingSet`, `BND.BindingSet` | MAPPED | Combined layout + binding set creation |
| `utils::ClearColorAttachment(ICommandList*, IFramebuffer*, uint32_t, Color)` | `UTL.ClearAttachments`, `GFX.Draw`, `STR.Soak` | MAPPED | Colour attachment clear helper |
| `utils::ClearDepthStencilAttachment(ICommandList*, IFramebuffer*, float, uint32_t)` | `UTL.ClearAttachments`, `TEX.DepthStencil` | MAPPED | Depth-stencil attachment clear helper |
| `utils::BuildBottomLevelAccelStruct(ICommandList*, rt::IAccelStruct*, const rt::AccelStructDesc&)` | `UTL.RayBuild` | GATED_OPTIONAL |  |
| `utils::TextureUavBarrier(ICommandList*, ITexture*)` | `UTL.Barriers`, `STA.Uav` | MAPPED | Texture UAV barrier helper |
| `utils::BufferUavBarrier(ICommandList*, IBuffer*)` | `UTL.Barriers`, `STA.Uav` | MAPPED | Buffer UAV barrier helper |
| `utils::ChooseFormat(IDevice*, FormatSupport, const Format*, size_t)` | `UTL.ChooseFormat`, `DEV.FormatSupport` | MAPPED | Ordered format capability selection |
| `utils::GraphicsAPIToString(GraphicsAPI)` | `UTL.Names`, `CPU.ObjectAndEnums`, `PKG.Build` | MAPPED | Valid and out-of-range enum strings |
| `utils::TextureDimensionToString(TextureDimension)` | `UTL.Names`, `CPU.ObjectAndEnums` | MAPPED | All texture dimensions + invalid fallback |
| `utils::DebugNameToString(const std::string&)` | `UTL.Names`, `CPU.ObjectAndEnums` | MAPPED | Empty (`<UNNAMED>`) vs non-empty strings |
| `utils::ShaderStageToString(ShaderType)` | `UTL.Names`, `CPU.ObjectAndEnums` | MAPPED | All shader stages + invalid fallback |
| `utils::ResourceTypeToString(ResourceType)` | `UTL.Names`, `CPU.ObjectAndEnums` | MAPPED | All resource types + `Count` fallback |
| `utils::FormatToString(Format)` | `UTL.Names`, `CPU.FormatMetadata` | MAPPED | Format string lookup |
| `utils::CommandQueueToString(CommandQueue)` | `UTL.Names`, `CPU.ObjectAndEnums` | MAPPED | Graphics, Compute, Copy, and `Count` |
| `utils::GenerateHeapDebugName(const HeapDesc&)` | `UTL.Names` | MAPPED | Non-exported declaration in `utils.h` |
| `utils::GenerateTextureDebugName(const TextureDesc&)` | `UTL.Names` | MAPPED | Non-exported declaration in `utils.h` |
| `utils::GenerateBufferDebugName(const BufferDesc&)` | `UTL.Names` | MAPPED | Non-exported declaration in `utils.h` |
| `utils::NotImplemented()` / `NotSupported()` / `InvalidEnum()` | `DEV.OptionalFailures` | MAPPED | Exercised via unsupported/invalid paths |
| `utils::BitSetAllocator` | `UTL.Allocator`, `CPU.Containers` | MAPPED | Single-threaded and multithreaded allocation/release |
| `utils::ScopedMarker` | `QRY.Markers` | MAPPED | RAII `beginMarker`/`endMarker` |

### `include/nvrhi/validation.h`

| Function | Test ID(s) | Status | Notes |
|---|---| --- | --- |
| `nvrhi::validation::createValidationLayer(IDevice*)` | `VAL.PassThrough`, `VAL.InvalidDescriptors`, `VAL.CommandOrder`, `VAL.BindingMismatch`, `VAL.QueueMismatch`, `VAL.ErrorCapture`, `VAL.OutputPreservation`, `VAL.NoFalseErrors` | MAPPED | Tested directly and via `--validation on |both` |

### `include/nvrhi/vulkan.h`

| Class / Function / Struct | Test ID(s) | Status | Notes |
|---|---| --- | --- |
| `nvrhi::vulkan::createDevice(const DeviceDesc&)` | `DEV.CreateDestroy`, `NAT.VulkanPipelineCache` | MAPPED | Standard and pipeline-cache-seeded device creation |
| `nvrhi::vulkan::convertFormat(nvrhi::Format)` | `NAT.VulkanSync` / `DEV.FormatSupport` | MAPPED | Format conversion verified on Vulkan |
| `nvrhi::vulkan::resultToString(VkResult)` | `NAT.VulkanSync` | MAPPED | VkResult diagnostic string mapping |
| `vulkan::IDevice::getQueueSemaphore(CommandQueue)` | `NAT.VulkanSync` | MAPPED | Timeline semaphore retrieval per queue |
| `vulkan::IDevice::queueWaitForSemaphore(CommandQueue, VkSemaphore, uint64_t)` | `NAT.VulkanSync` | MAPPED | GPU timeline semaphore wait |
| `vulkan::IDevice::queueSignalSemaphore(CommandQueue, VkSemaphore, uint64_t)` | `NAT.VulkanSync` | MAPPED | GPU timeline semaphore signal |
| `vulkan::IDevice::queueGetCompletedInstance(CommandQueue)` | `NAT.VulkanSync` | MAPPED | Monotonic completed timeline counter |
| `vulkan::IDevice::getPipelineCacheData(std::vector<uint8_t>&)` | `NAT.VulkanPipelineCache` | MAPPED | Exports pipeline cache blob |
| `vulkan::IDevice::getPipelineCacheDataSize()` | `NAT.VulkanPipelineCache` | MAPPED | Size-only query without full serialisation |
| `vulkan::IDevice::getMappedBufferFlushAlignment()` | `NAT.MappedFlush` | MAPPED | Non-zero alignment query |
| `vulkan::IDevice::flushMappedBuffer(IBuffer*, uint64_t, uint64_t)` | `NAT.MappedFlush` | MAPPED | Explicit mapped range flush + invalid-arg checks |
| `vulkan::DeviceDesc::pipelineCacheInitialData` / `Size` | `NAT.VulkanPipelineCache` | MAPPED | Seeds new device cache from exported blob |

### `include/nvrhi/metal.h`

| Class / Function | Test ID(s) | Status | Notes |
|---|---| --- | --- |
| `nvrhi::metal::createDevice(const DeviceDesc&)` | `DEV.CreateDestroy`, `NAT.Metal` | MAPPED | Default and explicit `MTLDevice` creation |
| `metal::IDevice::getNativeDevice()` | `NAT.Metal`, `NAT.Objects` | MAPPED | Borrowed `id<MTLDevice>` |
| `metal::IDevice::getNativeCommandQueue()` | `NAT.Metal`, `NAT.Objects` | MAPPED | Borrowed `id<MTLCommandQueue>` |
| `metal::IDevice::getNativeDeviceMemorySize()` | `NAT.Metal` | MAPPED | Recommended working set size > 0 |
| `metal::IDevice::getNativeAllocatedSize()` | `NAT.Metal` | MAPPED | Current allocated size query |
| `metal::IDevice::getMappedBufferFlushAlignment()` | `NAT.Metal`, `NAT.MappedFlush` | MAPPED | Mapped flush alignment query |
| `metal::IDevice::flushMappedBuffer(IBuffer*, uint64_t, uint64_t)` | `NAT.Metal`, `NAT.MappedFlush` | MAPPED | Managed/shared buffer range flush |

### `include/nvrhi/d3d11.h`

| Function | Test ID(s) | Status | Notes |
|---|---| --- | --- |
| `nvrhi::d3d11::createDevice(const DeviceDesc&)` | `DEV.CreateDestroy` | MAPPED |  |
| `nvrhi::d3d11::convertFormat(nvrhi::Format)` | — | GAP_UNIMPLEMENTED | No dedicated standalone UAA test is registered for this backend extension. |

### `include/nvrhi/d3d12.h`

| Class / Function | Test ID(s) | Status | Notes |
|---|---| --- | --- |
| `nvrhi::d3d12::createDevice(const DeviceDesc&)` | `DEV.CreateDestroy` | MAPPED |  |
| `nvrhi::d3d12::enableExperimentalFeatures(IMessageCallback*)` | — | GAP_UNIMPLEMENTED | No dedicated standalone UAA test is registered for this backend extension. |
| `nvrhi::d3d12::convertFormat(nvrhi::Format)` | — | GAP_UNIMPLEMENTED | No dedicated standalone UAA test is registered for this backend extension. |
| `d3d12::IDevice::buildRootSignature(...)` | — | GAP_UNIMPLEMENTED | No dedicated standalone UAA test is registered for this backend extension. |
| `d3d12::IDevice::createHandleForNativeGraphicsPipeline(...)` | — | GAP_UNIMPLEMENTED | No dedicated standalone UAA test is registered for this backend extension. |
| `d3d12::IDevice::createHandleForNativeMeshletPipeline(...)` | — | GAP_UNIMPLEMENTED | No dedicated standalone UAA test is registered for this backend extension. |
| `d3d12::IDevice::getDescriptorHeap(DescriptorHeapType)` | — | GAP_UNIMPLEMENTED | No dedicated standalone UAA test is registered for this backend extension. |
| `d3d12::ICommandList::allocateUploadBuffer(...)` | — | GAP_UNIMPLEMENTED | No dedicated standalone UAA test is registered for this backend extension. |
| `d3d12::ICommandList::commitDescriptorHeaps()` | — | GAP_UNIMPLEMENTED | No dedicated standalone UAA test is registered for this backend extension. |
| `d3d12::ICommandList::getBufferGpuVA(IBuffer*)` | — | GAP_UNIMPLEMENTED | No dedicated standalone UAA test is registered for this backend extension. |
| `d3d12::ICommandList::updateGraphicsVolatileBuffers()` | — | GAP_UNIMPLEMENTED | No dedicated standalone UAA test is registered for this backend extension. |
| `d3d12::ICommandList::updateComputeVolatileBuffers()` | — | GAP_UNIMPLEMENTED | No dedicated standalone UAA test is registered for this backend extension. |
| `d3d12::IDescriptorHeap::allocateDescriptors` / `allocateDescriptor` / `releaseDescriptors` / `releaseDescriptor` / `getCpuHandle` / `getCpuHandleShaderVisible` / `getGpuHandle` / `getHeap` / `getShaderVisibleHeap` | — | GAP_UNIMPLEMENTED | No dedicated standalone UAA test is registered for this backend extension. |

---

## 7. Key Descriptor Fields & UAA Extensions

| Descriptor / Field | Test ID(s) | Status | Notes |
|---|---| --- | --- |
| `ComponentMapping` (`r`, `g`, `b`, `a`, `pack`, `unpack`, setters) | `CPU.DescriptorDefaults`, `TEX.Views` | MAPPED | Identity and arbitrary swizzles (`Zero`, `One`, channel swaps) |
| `TextureDesc::defaultComponentMapping` | `TEX.Views`, `BND.BindingSet` | MAPPED | Inherited when SRV binding does not override |
| `BindingSetItem::overrideComponentMapping` / `setOverrideComponentMapping(...)` | `BND.BindingSet`, `TEX.Views`, `NAT.NativeViews` | MAPPED | Explicit identity override vs `std::nullopt` inheritance |
| `TextureDesc` (`width`, `height`, `depth`, `arraySize`, `mipLevels`, `sampleCount`, `format`, `dimension`, `isRenderTarget`, `isUAV`, `isTypeless`, `isVirtual`, `isTiled`, `isShadingRateSurface`, `clearValue`, `initialState`, `keepInitialState`, `sharedResourceFlags`) | `CPU.DescriptorDefaults`, `DEV.ResourceDescriptions`, `TEX.*` | MAPPED | Builder defaults and runtime creation verified |
| `BufferDesc` (`byteSize`, `structStride`, `maxVersions`, `format`, `canHaveUAVs`, `canHaveTypedViews`, `canHaveRawViews`, `isVertexBuffer`, `isIndexBuffer`, `isConstantBuffer`, `isDrawIndirectArgs`, `isAccelStructBuildInput`, `isAccelStructStorage`, `isShaderBindingTable`, `isVolatile`, `isVirtual`, `cpuAccess`, `initialState`, `keepInitialState`) | `CPU.DescriptorDefaults`, `DEV.ResourceDescriptions`, `BUF.*`, `RT.*` | MAPPED | All buffer usage flags covered |
| `BindlessLayoutDesc::updateAfterBind` / `setUpdateAfterBind(bool)` | `TBL.UpdateAfterBind`, `CPU.DescriptorDefaults` | GATED_OPTIONAL | Verifies default `true` and opt-out `false` behaviour on Vulkan/bindless backends |
| `BindlessLayoutDesc::layoutType` (`Immutable`, `MutableSrvUavCbv`, `MutableCounters`, `MutableSampler`) | `TBL.CreateWrite`; D3D12 heap-index case is a gap | GATED_OPTIONAL |  |
| `SamplerDesc::compareFunc` / `setCompareFunc(ComparisonFunc)` | `SMP.Options`, `CPU.DescriptorDefaults` | MAPPED | Comparison sampling with `cs_sampler_compare` |
| `SamplerDesc::reductionType` (`Standard`, `Comparison`, `Minimum`, `Maximum`) | `SMP.Options`, `DEV.ResourceDescriptions` | MAPPED | Standard and comparison tested; Min/Max checked where supported |
| `VulkanBindingOffsets` (`shaderResource`, `sampler`, `constantBuffer`, `unorderedAccess`) | `CPU.DescriptorDefaults`, `BND.Layout`, `BND.StageIsolation` | MAPPED | Explicit zero-offset and custom-offset Vulkan layouts |
| `BindingLayoutDesc::registerSpace` / `registerSpaceIsDescriptorSet` | `BND.Layout`, `BND.StageIsolation`, `TBL.CreateWrite` | MAPPED | Multi-set Vulkan descriptor sets and DX12 spaces |
| `BindingSetDesc::trackLiveness` | `BND.BindingLifetime`, `SUB.LifetimeTracker` | MAPPED | Automatic vs manual liveness tracking |
| `CommandListParameters` (`enableImmediateExecution`, `uploadChunkSize`, `scratchChunkSize`, `scratchMaxMemory`, `queueType`, `lifetimeTracker`) | `SUB.ListLifecycle`, `SUB.QueueDependencies`, `SUB.LifetimeTracker`, `STR.Threads` | MAPPED | Core lifecycle and queue fields are exercised; every tuning field still needs field-specific verification. |
