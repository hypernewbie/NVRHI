# NVRHI standalone API tests

<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. MIT licence, see ../../LICENSE.txt. -->

The suite exercises NVRHI directly. The CPU executable requires no graphics device. GPU tests use deterministic inputs and inspect readback through NVRHI. See the [API coverage manifest](API_COVERAGE_MANIFEST.md) for source mappings and gaps.

## Build and run

From the NVRHI source directory, configure with `NVRHI_BUILD_TESTS=ON` and enable the backends to be tested.

```sh
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DNVRHI_BUILD_TESTS=ON
cmake --build build-debug --config Debug --parallel
ctest --test-dir build-debug -C Debug -N
ctest --test-dir build-debug -C Debug -L uaa --output-on-failure
```

CTest registers CPU, package, and separate raw, validation, shader-rejection and stress invocations for each enabled backend. The validation invocation requests both NVRHI and native validation. CTest uses exit code 77 for an unavailable invocation. Individual skips and their reasons appear in runner output and in reports. The legacy memory-query and Metal framebuffer targets remain available under their existing CTest names.

For a single-config build, use the runner directly to list backend cases, filter a test, or save structured results.

```sh
build-debug/tests/uaa/nvrhi_uaa_tests --backend vulkan --list
build-debug/tests/uaa/nvrhi_uaa_tests --backend vulkan --filter CMP.Direct --report compute.json
MTL_DEBUG_LAYER=1 build-debug/tests/uaa/nvrhi_uaa_tests --backend metal --validation on --native-validation on --filter CMP.Direct --report metal.json
```

Select Vulkan implementations through the loader environment, for example by setting `VK_ICD_FILENAMES` to one ICD manifest. Use `--device` to select an enumerated adapter by index or name. The `--seed` and `--timeout` options control deterministic inputs and the per-case watchdog. Use `--help` for the full list.

If a driver aborts, rerun the affected case with `--filter <id>` and keep its log with the signal or exit status. Each invocation writes its JSON report at completion. The test process must survive to write that report. The standalone shader-rejection CTest invocation keeps deliberately invalid shader input out of the core runs.

## Shader fixtures

The bundled SPIR-V and DXIL binaries allow test execution without shader tools. `shaders/manifest.json` records artifact hashes, compiler information, profiles and flags. To rebuild the regular fixtures, make DXC, `spirv-val`, `spirv-link`, `spirv-dis` and `spirv-cross` available in PATH or under `VULKAN_SDK/bin`, then build the `nvrhi_uaa_compile_shaders` target. The generator retains the checked-in Metal SPIR-V fixtures unless invoked with `--update-frozen`.

D3D11 needs Shader Model 5 DXBC. From a Windows SDK environment, run `python shaders/compile_shaders.py --fxc <path-to-fxc>` from `tests/uaa`. The FXC path has yet to be verified on Windows.

## Package checks

`uaa_package` builds and runs a consumer against the current library. With install rules enabled on Unix, it stages the headers and library using `DESTDIR` and checks that the UAA executables are absent from the installation. Windows currently uses build-tree inputs because the fork's header install destination is absolute. The fork's static package export and shared Metal build have upstream CMake target issues, independent of the UAA targets.
