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
