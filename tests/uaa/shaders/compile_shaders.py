#!/usr/bin/env python3
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Compiles UAA shaders to SPIR-V (and HLSL to DXIL if supported), validates
them with spirv-val and spirv-cross MSL translation, freezes Metal SPIR-V
fixtures, and writes shaders/manifest.json."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def find_tool(name: str) -> str:
    """Locate a shader toolchain binary via PATH or VULKAN_SDK/bin."""
    found = shutil.which(name)
    if found:
        return found
    vulkan_sdk = os.environ.get("VULKAN_SDK")
    if vulkan_sdk:
        candidate = Path(vulkan_sdk) / "bin" / name
        if candidate.is_file():
            return str(candidate)
        candidate_exe = Path(vulkan_sdk) / "bin" / f"{name}.exe"
        if candidate_exe.is_file():
            return str(candidate_exe)
    raise RuntimeError(f"Required tool '{name}' not found in PATH or VULKAN_SDK/bin")


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Execute a subprocess and capture stdout/stderr as text."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def probe_dxil_support(dxc: str, hlsl_dir: Path) -> bool:
    """Check whether local dxc can emit DXIL binaries."""
    probe_src = hlsl_dir / "cs_push_constants.hlsl"
    with tempfile.TemporaryDirectory() as tmp:
        out_dxil = Path(tmp) / "probe.dxil"
        proc = run_cmd(
            [dxc, "-T", "cs_6_0", "-E", "main", "-Fo", str(out_dxil), str(probe_src)],
            check=False,
        )
        if proc.returncode == 0 and out_dxil.is_file() and out_dxil.stat().st_size > 0:
            return True
        print(f"[INFO] DXIL compilation not supported by local dxc (skipping DXIL): {proc.stderr.strip()}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fxc", help="Path to the Windows SDK FXC executable for D3D11 fixtures")
    parser.add_argument("--update-frozen", action="store_true", help="Replace the checked-in Metal SPIR-V fixtures")
    args = parser.parse_args()
    shaders_dir = Path(__file__).resolve().parent
    hlsl_dir = shaders_dir / "hlsl"
    spirv_dir = shaders_dir / "bin" / "spirv"
    dxil_dir = shaders_dir / "bin" / "dxil"
    frozen_metal_dir = shaders_dir / "frozen_metal"

    spirv_dir.mkdir(parents=True, exist_ok=True)
    frozen_metal_dir.mkdir(parents=True, exist_ok=True)

    dxc = find_tool("dxc")
    spirv_link = find_tool("spirv-link")
    spirv_val = find_tool("spirv-val")
    spirv_cross = find_tool("spirv-cross")
    spirv_dis = find_tool("spirv-dis")
    glslang = find_tool("glslangValidator")

    ver_proc = run_cmd([dxc, "--version"], check=False)
    compiler_version = (ver_proc.stdout or ver_proc.stderr).strip().splitlines()[0] if (ver_proc.stdout or ver_proc.stderr) else "unknown"

    # Shader specification list:
    # (name, source, entry, target, extra_spv_flags, test_msl, val_env)
    single_shaders = [
        ("cs_buffer_copy_add", "cs_buffer_copy_add.hlsl", "main", "cs_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("cs_raw_typed_rw", "cs_raw_typed_rw.hlsl", "main", "cs_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("cs_push_constants", "cs_push_constants.hlsl", "main", "cs_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("cs_spec_const", "cs_spec_const.hlsl", "main", "cs_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("cs_indirect_gen", "cs_indirect_gen.hlsl", "main", "cs_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("cs_indirect_work", "cs_indirect_work.hlsl", "main", "cs_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("cs_sampler_test", "cs_sampler_test.hlsl", "main", "cs_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("cs_sampler_compare", "cs_sampler_compare.hlsl", "main", "cs_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("cs_texture_uav_fill", "cs_texture_uav_fill.hlsl", "main", "cs_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("cs_uav_chain", "cs_uav_chain.hlsl", "main", "cs_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("cs_bindings_all", "cs_bindings_all.hlsl", "main", "cs_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("cs_multi_set", "cs_multi_set.hlsl", "main", "cs_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("cs_bindless_tex_buf", "cs_bindless_tex_buf.hlsl", "main", "cs_6_0", ["-fspv-target-env=vulkan1.2"], "bindless_msl", "vulkan1.2"),
        ("cs_bindless_sampler", "cs_bindless_sampler.hlsl", "main", "cs_6_0", ["-fspv-target-env=vulkan1.2"], "bindless_msl", "vulkan1.2"),
        ("cs_bindless_typed_uav", "cs_bindless_typed_uav.hlsl", "main", "cs_6_0", ["-fspv-target-env=vulkan1.2"], "bindless_msl", "vulkan1.2"),
        ("cs_bindless_two_tex_kinds", "cs_bindless_two_tex_kinds.hlsl", "main", "cs_6_0", ["-fspv-target-env=vulkan1.2"], "bindless_msl", "vulkan1.2"),
        ("cs_bindless_cube_sparse", "cs_bindless_cube_sparse.hlsl", "main", "cs_6_0", ["-fspv-target-env=vulkan1.2"], "bindless_msl", "vulkan1.2"),
        ("cs_bindless_uint_sparse", "cs_bindless_uint_sparse.hlsl", "main", "cs_6_0", ["-fspv-target-env=vulkan1.2"], "bindless_msl", "vulkan1.2"),
        ("vs_bindless_stage", "vs_bindless_stage.hlsl", "main", "vs_6_0", ["-fspv-target-env=vulkan1.2"], "bindless_msl", "vulkan1.2"),
        ("ps_bindless_stage", "ps_bindless_stage.hlsl", "main", "ps_6_0", ["-fspv-target-env=vulkan1.2"], "bindless_msl", "vulkan1.2"),
        ("cs_wave_lane", "cs_wave_lane.hlsl", "main", "cs_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("vs_fullscreen_tri", "vs_fullscreen_tri.hlsl", "main", "vs_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("ps_solid_color", "ps_solid_color.hlsl", "main", "ps_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("ps_mrt", "ps_mrt.hlsl", "main", "ps_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("vs_vertex_input", "vs_vertex_input.hlsl", "main", "vs_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("ps_vertex_color", "ps_vertex_color.hlsl", "main", "ps_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("vs_stage_iso", "vs_stage_iso.hlsl", "main", "vs_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("ps_stage_iso", "ps_stage_iso.hlsl", "main", "ps_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("cs_metal_custom_entry", "cs_metal_custom_entry.hlsl", "uaa_custom_entry", "cs_6_0", ["-fspv-target-env=vulkan1.1"], True, "vulkan1.1"),
        ("cs_ray_query", "cs_ray_query.hlsl", "main", "cs_6_5", ["-fspv-target-env=vulkan1.2", "-fspv-extension=SPV_KHR_ray_query"], False, "vulkan1.2"),
        ("ms_triangle", "ms_triangle.hlsl", "main", "ms_6_5", ["-fspv-target-env=vulkan1.3", "-fspv-extension=SPV_EXT_mesh_shader"], False, "vulkan1.3"),
        ("ms_triangle_payload", "ms_triangle_payload.hlsl", "main", "ms_6_5", ["-fspv-target-env=vulkan1.3", "-fspv-extension=SPV_EXT_mesh_shader"], False, "vulkan1.3"),
        ("as_task", "as_task.hlsl", "main", "as_6_5", ["-fspv-target-env=vulkan1.3", "-fspv-extension=SPV_EXT_mesh_shader"], False, "vulkan1.3"),
        ("rt_lib", "rt_lib.hlsl", "", "lib_6_3", ["-fspv-target-env=vulkan1.2", "-fspv-extension=SPV_KHR_ray_tracing"], False, "vulkan1.2"),
    ]

    frozen_metal_names = {
        "cs_push_constants",
        "cs_sampler_test",
        "cs_raw_typed_rw",
        "cs_bindings_all",
        "cs_multi_set",
        "cs_spec_group_size",
        "cs_bindless_tex_buf",
        "cs_bindless_sampler",
        "cs_bindless_typed_uav",
        "cs_bindless_two_tex_kinds",
        "cs_bindless_cube_sparse",
        "cs_bindless_uint_sparse",
        "vs_bindless_stage",
        "ps_bindless_stage",
        "cs_metal_custom_entry",
    }

    artifacts = []
    fxc = args.fxc or shutil.which("fxc")
    dxbc_dir = shaders_dir / "bin" / "dxbc"
    if fxc:
        dxbc_dir.mkdir(parents=True, exist_ok=True)
        fxc_help = run_cmd([fxc, "/?"], check=False)
        fxc_version = next((line.strip() for line in (fxc_help.stdout + fxc_help.stderr).splitlines()
                            if "Shader Compiler" in line), "unknown")
    dxil_supported = probe_dxil_support(dxc, hlsl_dir)
    if dxil_supported:
        dxil_dir.mkdir(parents=True, exist_ok=True)

    base_spv_flags = ["-spirv", "-fvk-use-dx-layout"]

    for name, src_name, entry, target, extra_flags, test_msl, val_env in single_shaders:
        src_path = hlsl_dir / src_name
        spv_path = spirv_dir / f"{name}.spv"

        flags = list(base_spv_flags) + list(extra_flags)
        cmd = [dxc, "-T", target]
        if entry:
            cmd += ["-E", entry]
        cmd += flags + ["-Fo", str(spv_path), str(src_path)]

        run_cmd(cmd)
        run_cmd([spirv_val, "--target-env", val_env, str(spv_path)])

        if test_msl:
            msl_cmd = [spirv_cross, "--msl", "--msl-version", "30000", "--msl-texture-buffer-native"]
            if test_msl == "bindless_msl":
                msl_cmd += ["--msl-argument-buffers", "--msl-argument-buffer-tier", "1", "--msl-device-argument-buffer", "0"]
            if entry:
                msl_cmd += ["--entry", entry]
            msl_cmd.append(str(spv_path))
            run_cmd(msl_cmd)

        print(f"[OK] spirv: {name}.spv ({spv_path.stat().st_size} bytes)")
        artifacts.append({
            "name": name,
            "kind": "spirv",
            "source": f"hlsl/{src_name}",
            "entry": entry if entry else "RayGen,Miss,ClosestHit,AnyHit",
            "target": target,
            "flags": flags,
            "output": f"bin/spirv/{name}.spv",
            "bytes": spv_path.stat().st_size,
            "sha256": sha256_file(spv_path),
        })

        if dxil_supported:
            dxil_path = dxil_dir / f"{name}.dxil"
            dxil_cmd = [dxc, "-T", target]
            if entry:
                dxil_cmd += ["-E", entry]
            dxil_cmd += ["-Fo", str(dxil_path), str(src_path)]
            run_cmd(dxil_cmd)
            print(f"[OK] dxil:  {name}.dxil ({dxil_path.stat().st_size} bytes)")
            artifacts.append({
                "name": name,
                "kind": "dxil",
                "source": f"hlsl/{src_name}",
                "entry": entry if entry else "RayGen,Miss,ClosestHit,AnyHit",
                "target": target,
                "flags": [],
                "output": f"bin/dxil/{name}.dxil",
                "bytes": dxil_path.stat().st_size,
                "sha256": sha256_file(dxil_path),
            })

        if fxc and target in ("cs_6_0", "vs_6_0", "ps_6_0") and name not in {
            "cs_spec_const", "cs_multi_set", "cs_bindless_tex_buf", "cs_wave_lane", "cs_metal_custom_entry",
            "cs_bindless_sampler", "cs_bindless_typed_uav", "cs_bindless_two_tex_kinds", "cs_bindless_cube_sparse", "cs_bindless_uint_sparse", "vs_bindless_stage", "ps_bindless_stage",
        }:
            dxbc_path = dxbc_dir / f"{name}.dxbc"
            dxbc_target = target[:2] + "_5_0"
            source = src_path.read_text(encoding="utf-8")
            source = re.sub(r"\[\[vk::[^\]]+\]\]\s*", "", source)
            source = re.sub(r",\s*space\d+\)", ")", source)
            source = re.sub(
                r"ConstantBuffer<(\w+)>\s+(\w+)\s*:\s*register\((b\d+)\);",
                r"cbuffer UaaPushConstants : register(\3) { \1 \2; };",
                source,
            )
            with tempfile.TemporaryDirectory() as scratch:
                fxc_source = Path(scratch) / src_name
                fxc_source.write_text(source, encoding="utf-8")
                run_cmd([fxc, "/nologo", "/T", dxbc_target, "/E", entry, "/Fo", str(dxbc_path), str(fxc_source)])
            artifacts.append({
                "name": name,
                "kind": "dxbc",
                "compiler": "fxc",
                "compilerVersion": fxc_version,
                "source": f"hlsl/{src_name}",
                "entry": entry,
                "target": dxbc_target,
                "flags": ["/nologo"],
                "output": f"bin/dxbc/{name}.dxbc",
                "bytes": dxbc_path.stat().st_size,
                "sha256": sha256_file(dxbc_path),
            })

    # GLSL supplies a workgroup-size specialisation fixture.
    group_src = shaders_dir / "glsl" / "cs_spec_group_size.comp"
    group_spv = spirv_dir / "cs_spec_group_size.spv"
    group_flags = ["-V", "--target-env", "vulkan1.1"]
    run_cmd([glslang] + group_flags + ["-o", str(group_spv), str(group_src)])
    run_cmd([spirv_val, "--target-env", "vulkan1.1", str(group_spv)])
    run_cmd([spirv_cross, "--msl", "--msl-version", "30000", "--entry", "main", str(group_spv)])
    print(f"[OK] spirv: cs_spec_group_size.spv ({group_spv.stat().st_size} bytes)")
    artifacts.append({
        "name": "cs_spec_group_size",
        "kind": "spirv",
        "compiler": "glslangValidator",
        "source": "glsl/cs_spec_group_size.comp",
        "entry": "main",
        "target": "vulkan1.1",
        "flags": group_flags,
        "output": "bin/spirv/cs_spec_group_size.spv",
        "bytes": group_spv.stat().st_size,
        "sha256": sha256_file(group_spv),
    })

    # Compile and link multi-entry compute shader module.
    multi_src = hlsl_dir / "cs_multi_entry.hlsl"
    multi_spv = spirv_dir / "cs_multi_entry.spv"
    multi_flags = list(base_spv_flags) + ["-fspv-target-env=vulkan1.1"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        add_spv = tmp_dir / "entry_add.spv"
        mul_spv = tmp_dir / "entry_mul.spv"

        run_cmd([dxc, "-T", "cs_6_0", "-E", "entry_add"] + multi_flags + ["-Fo", str(add_spv), str(multi_src)])
        run_cmd([dxc, "-T", "cs_6_0", "-E", "entry_mul"] + multi_flags + ["-Fo", str(mul_spv), str(multi_src)])
        run_cmd([spirv_link, "-o", str(multi_spv), str(add_spv), str(mul_spv)])

    run_cmd([spirv_val, "--target-env", "vulkan1.1", str(multi_spv)])

    dis_proc = run_cmd([spirv_dis, "--no-color", str(multi_spv)])
    if 'OpEntryPoint GLCompute %entry_add "entry_add"' not in dis_proc.stdout and '"entry_add"' not in dis_proc.stdout:
        raise RuntimeError("cs_multi_entry.spv missing entry_add OpEntryPoint")
    if '"entry_mul"' not in dis_proc.stdout:
        raise RuntimeError("cs_multi_entry.spv missing entry_mul OpEntryPoint")

    for entry_name in ("entry_add", "entry_mul"):
        run_cmd([
            spirv_cross, "--msl", "--msl-version", "30000",
            "--msl-texture-buffer-native", "--entry", entry_name, str(multi_spv)
        ])

    print(f"[OK] spirv: cs_multi_entry.spv ({multi_spv.stat().st_size} bytes, multi-entry)")
    artifacts.append({
        "name": "cs_multi_entry",
        "kind": "spirv",
        "source": "hlsl/cs_multi_entry.hlsl",
        "entry": "entry_add,entry_mul",
        "target": "cs_6_0",
        "flags": multi_flags + ["spirv-link"],
        "output": "bin/spirv/cs_multi_entry.spv",
        "bytes": multi_spv.stat().st_size,
        "sha256": sha256_file(multi_spv),
    })

    # Freeze Metal-compatible SPIR-V fixtures.
    for name in sorted(frozen_metal_names):
        src_spv = spirv_dir / f"{name}.spv"
        dst_spv = frozen_metal_dir / f"{name}.spv"
        if args.update_frozen:
            shutil.copy2(src_spv, dst_spv)
        elif not dst_spv.is_file():
            raise RuntimeError(f"Missing checked-in fixture {dst_spv}; use --update-frozen to recreate it")
        entry_val = "uaa_custom_entry" if name == "cs_metal_custom_entry" else "main"
        print(f"[OK] frozen_metal: {name}.spv ({dst_spv.stat().st_size} bytes)")
        artifacts.append({
            "name": name,
            "kind": "frozen_metal",
            "source": f"bin/spirv/{name}.spv",
            "entry": entry_val,
            "target": "vulkan1.1" if name == "cs_spec_group_size" else "cs_6_0",
            "flags": [],
            "output": f"frozen_metal/{name}.spv",
            "bytes": dst_spv.stat().st_size,
            "sha256": sha256_file(dst_spv),
        })

    manifest = {
        "compiler": "dxc",
        "compilerVersion": compiler_version,
        "licence": "MIT",
        "dxilSupported": dxil_supported,
        "artifacts": artifacts,
    }

    manifest_path = shaders_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"[DONE] Wrote {manifest_path} with {len(artifacts)} artifacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
