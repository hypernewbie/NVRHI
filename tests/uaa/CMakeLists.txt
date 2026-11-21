#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

set(_uaa_shader_dir "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(_uaa_build_config "$<IF:$<BOOL:$<CONFIG>>,$<CONFIG>,${CMAKE_BUILD_TYPE}>")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/uaa_cpu.cpp")
    add_executable(nvrhi_uaa_cpu
        uaa_main.cpp
        uaa_cpu.cpp)
    target_link_libraries(nvrhi_uaa_cpu PRIVATE nvrhi)
    target_compile_definitions(nvrhi_uaa_cpu PRIVATE
        UAA_CPU_ONLY=1
        UAA_NVRHI_SHARED=$<BOOL:${NVRHI_BUILD_SHARED}>
        UAA_WITH_VALIDATION=$<BOOL:${NVRHI_WITH_VALIDATION}>
        UAA_WITH_VULKAN=0
        UAA_WITH_METAL=0
        UAA_WITH_DX11=0
        UAA_WITH_D3D11=0
        UAA_WITH_DX12=0
        UAA_WITH_D3D12=0
        UAA_SHADER_DIR="${_uaa_shader_dir}"
        UAA_BUILD_CONFIG="${_uaa_build_config}")
    set_target_properties(nvrhi_uaa_cpu PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        FOLDER "NVRHI/Tests")

    add_test(NAME uaa_cpu COMMAND nvrhi_uaa_cpu --backend cpu)
    set_tests_properties(uaa_cpu PROPERTIES
        LABELS "uaa;cpu"
        TIMEOUT 60
        SKIP_RETURN_CODE 77)
endif()

add_executable(nvrhi_uaa_tests
    uaa_bindings.cpp
    uaa_bindless.cpp
    uaa_buffer.cpp
    uaa_compute.cpp
    uaa_coopvec.cpp
    uaa_cpu.cpp
    uaa_device.cpp
    uaa_fixture.cpp
    uaa_graphics.cpp
    uaa_heaps.cpp
    uaa_main.cpp
    uaa_mesh.cpp
    uaa_native.cpp
    uaa_queries.cpp
    uaa_raytracing.cpp
    uaa_rt_advanced.cpp
    uaa_sampler.cpp
    uaa_shader.cpp
    uaa_sparse.cpp
    uaa_states.cpp
    uaa_stress.cpp
    uaa_submission.cpp
    uaa_texture.cpp
    uaa_utils.cpp
    uaa_validation.cpp)

target_link_libraries(nvrhi_uaa_tests PRIVATE nvrhi)

set(_uaa_has_vulkan 0)
if(NVRHI_WITH_VULKAN)
    set(_uaa_has_vulkan 1)
    target_sources(nvrhi_uaa_tests PRIVATE uaa_vulkan.cpp)
    target_link_libraries(nvrhi_uaa_tests PRIVATE
        ${nvrhi_vulkan_target}
        ${nvrhi_vulkan_headers_target}
        ${CMAKE_DL_LIBS})
endif()

set(_uaa_has_metal 0)
if(NVRHI_WITH_METAL)
    set(_uaa_has_metal 1)
    target_sources(nvrhi_uaa_tests PRIVATE uaa_metal.mm)
    target_link_libraries(nvrhi_uaa_tests PRIVATE ${nvrhi_metal_target})
    if(NVRHI_METAL_TEST_HOOKS)
        target_sources(nvrhi_uaa_tests PRIVATE uaa_metal_submission.mm)
        target_compile_definitions(nvrhi_uaa_tests PRIVATE NVRHI_METAL_TEST_HOOKS=1)
        set_source_files_properties(uaa_metal_submission.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    endif()
    if(APPLE)
        set_source_files_properties(uaa_metal.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    endif()
endif()

set(_uaa_has_dx11 0)
if(NVRHI_WITH_DX11)
    set(_uaa_has_dx11 1)
    target_sources(nvrhi_uaa_tests PRIVATE uaa_d3d11.cpp)
    target_link_libraries(nvrhi_uaa_tests PRIVATE ${nvrhi_d3d11_target})
endif()

set(_uaa_has_dx12 0)
if(NVRHI_WITH_DX12)
    set(_uaa_has_dx12 1)
    target_sources(nvrhi_uaa_tests PRIVATE uaa_d3d12.cpp)
    target_link_libraries(nvrhi_uaa_tests PRIVATE ${nvrhi_d3d12_target})
endif()

target_compile_definitions(nvrhi_uaa_tests PRIVATE
    UAA_CPU_ONLY=0
    UAA_NVRHI_SHARED=$<BOOL:${NVRHI_BUILD_SHARED}>
    UAA_WITH_VALIDATION=$<BOOL:${NVRHI_WITH_VALIDATION}>
    UAA_WITH_VULKAN=${_uaa_has_vulkan}
    UAA_WITH_METAL=${_uaa_has_metal}
    UAA_WITH_DX11=${_uaa_has_dx11}
    UAA_WITH_D3D11=${_uaa_has_dx11}
    UAA_WITH_DX12=${_uaa_has_dx12}
    UAA_WITH_D3D12=${_uaa_has_dx12}
    UAA_SHADER_DIR="${_uaa_shader_dir}"
    UAA_NVRHI_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    UAA_NVRHI_BINARY_DIR="${PROJECT_BINARY_DIR}"
    UAA_BUILD_CONFIG="${_uaa_build_config}")

set_target_properties(nvrhi_uaa_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    FOLDER "NVRHI/Tests")

function(nvrhi_uaa_register_backend_tests backend_name)
    add_test(
        NAME "uaa_${backend_name}_raw"
        COMMAND nvrhi_uaa_tests
            --backend "${backend_name}"
            --validation off
            --exclude-tag stress
            --exclude-tag destructive
            --exclude-tag package
            --exclude-tag isolated)
    set_tests_properties("uaa_${backend_name}_raw" PROPERTIES
        LABELS "uaa;${backend_name};raw"
        TIMEOUT 180
        SKIP_RETURN_CODE 77)

    if(NVRHI_WITH_VALIDATION)
        set(_uaa_native_val_flag --native-validation on)
        add_test(
            NAME "uaa_${backend_name}_validation"
            COMMAND nvrhi_uaa_tests
                --backend "${backend_name}"
                --validation on
                ${_uaa_native_val_flag}
                --exclude-tag stress
                --exclude-tag destructive
                --exclude-tag package
            --exclude-tag isolated)
        set_tests_properties("uaa_${backend_name}_validation" PROPERTIES
            LABELS "uaa;${backend_name};validation"
            TIMEOUT 180
            SKIP_RETURN_CODE 77)
        if("${backend_name}" STREQUAL "metal")
            set_tests_properties("uaa_${backend_name}_validation" PROPERTIES
                ENVIRONMENT "MTL_DEBUG_LAYER=1")
        endif()
    endif()

    add_test(
        NAME "uaa_${backend_name}_shader_rejection"
        COMMAND nvrhi_uaa_tests
            --backend "${backend_name}"
            --validation off
            --filter SHD.InvalidBinary)
    set_tests_properties("uaa_${backend_name}_shader_rejection" PROPERTIES
        LABELS "uaa;${backend_name};isolated"
        TIMEOUT 60
        SKIP_RETURN_CODE 77)

    add_test(
        NAME "uaa_${backend_name}_stress"
        COMMAND nvrhi_uaa_tests
            --backend "${backend_name}"
            --validation off
            --tag stress)
    set_tests_properties("uaa_${backend_name}_stress" PROPERTIES
        LABELS "uaa;${backend_name};stress"
        TIMEOUT 300
        SKIP_RETURN_CODE 77)
endfunction()

if(NVRHI_WITH_METAL)
    nvrhi_uaa_register_backend_tests(metal)
    add_test(NAME uaa_metal_teardown_raw
        COMMAND nvrhi_uaa_tests --backend metal --validation off --filter SUB.MetalTeardown --timeout 15)
    set_tests_properties(uaa_metal_teardown_raw PROPERTIES
        LABELS "uaa;metal;isolated" TIMEOUT 25 SKIP_RETURN_CODE 77
        ENVIRONMENT "MTL_DEBUG_LAYER=1")
    if(NVRHI_WITH_VALIDATION)
        add_test(NAME uaa_metal_teardown_validation
            COMMAND nvrhi_uaa_tests --backend metal --validation on --native-validation on --filter SUB.MetalTeardown --timeout 15)
        set_tests_properties(uaa_metal_teardown_validation PROPERTIES
            LABELS "uaa;metal;isolated;validation" TIMEOUT 25 SKIP_RETURN_CODE 77
            ENVIRONMENT "MTL_DEBUG_LAYER=1")
    endif()
    if(NVRHI_METAL_TEST_HOOKS)
        add_test(NAME uaa_metal_submission_raw
            COMMAND nvrhi_uaa_tests --backend metal --validation off --filter SUB.MetalHook* --timeout 15)
        set_tests_properties(uaa_metal_submission_raw PROPERTIES
            LABELS "uaa;metal;isolated" TIMEOUT 50 SKIP_RETURN_CODE 77
            ENVIRONMENT "MTL_DEBUG_LAYER=1")
        if(NVRHI_WITH_VALIDATION)
            add_test(NAME uaa_metal_submission_validation
                COMMAND nvrhi_uaa_tests --backend metal --validation on --native-validation on --filter SUB.MetalHook* --timeout 15)
            set_tests_properties(uaa_metal_submission_validation PROPERTIES
                LABELS "uaa;metal;isolated;validation" TIMEOUT 50 SKIP_RETURN_CODE 77
                ENVIRONMENT "MTL_DEBUG_LAYER=1")
        endif()
    endif()
endif()

if(NVRHI_WITH_VULKAN)
    nvrhi_uaa_register_backend_tests(vulkan)
endif()

if(NVRHI_WITH_DX11)
    nvrhi_uaa_register_backend_tests(d3d11)
endif()

if(NVRHI_WITH_DX12)
    nvrhi_uaa_register_backend_tests(d3d12)
endif()

add_test(
    NAME uaa_package
    COMMAND nvrhi_uaa_tests --backend cpu --tag package --timeout 150)
set_tests_properties(uaa_package PROPERTIES
    LABELS "uaa;package"
    TIMEOUT 180
    SKIP_RETURN_CODE 77)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
    set(_uaa_python "${Python3_EXECUTABLE}")
else()
    set(_uaa_python "python3")
endif()

add_custom_target(nvrhi_uaa_compile_shaders
    COMMAND "${_uaa_python}" "${_uaa_shader_dir}/compile_shaders.py"
    WORKING_DIRECTORY "${_uaa_shader_dir}"
    COMMENT "Compiling NVRHI UAA test shaders")
