"""Setup file retained for editable installs and CUDA extension build.

Most metadata is declared in pyproject.toml (PEP 621). We keep this file so
`pip install .` still builds the optional CUDA extension. Importing torch is
done lazily so that metadata queries (e.g., pip’s build isolation step) do not
fail when torch is not pre-installed.

Environment variables:
  DANCINGMODEL_SKIP_EXT=1   -> skip building the CUDA extension entirely.
  DANCINGMODEL_VERBOSE=1    -> print debug info about extension discovery.
"""

from __future__ import annotations

import os
import pathlib
from setuptools import setup, find_packages

ROOT = pathlib.Path(__file__).parent


def _glob(pattern: str):
  # Return relative POSIX paths (setuptools requires forward slashes, relative)
  out = []
  for p in ROOT.glob(pattern):
    if p.is_file():
      out.append(p.relative_to(ROOT).as_posix())
  return out


def _try_import_torch_extension():
  try:
    import torch.utils.cpp_extension as torch_cpp_ext  # type: ignore
  except ModuleNotFoundError as e:
    raise RuntimeError(
      "PyTorch is required to build the CUDA extension. Install torch first, "
      "or set DANCINGMODEL_SKIP_EXT=1 to skip building native kernels.\n"
      "Example: pip install 'dancingmodel[torch]'"
    ) from e
  return torch_cpp_ext


def _remove_unwanted_pytorch_nvcc_flags(torch_cpp_ext):
  remove = [
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
  ]
  for flag in remove:
    try:
      torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
    except ValueError:
      pass


def _build_ext_modules():
  if os.environ.get("DANCINGMODEL_SKIP_EXT"):
    if os.environ.get("DANCINGMODEL_VERBOSE"):
      print("[dancingmodel] Skipping CUDA extension build (env var set).")
    return []

  torch_cpp_ext = _try_import_torch_extension()
  _remove_unwanted_pytorch_nvcc_flags(torch_cpp_ext)

  sources = ["dancingmodel/csrc/lora_ops.cc"] + _glob("dancingmodel/csrc/bgmv/*.cu")
  # Defensive: ensure all are relative (no absolute paths)
  sources = [pathlib.Path(s).as_posix() if not pathlib.Path(s).is_absolute() else pathlib.Path(s).relative_to(ROOT).as_posix() for s in sources]
  if os.environ.get("DANCINGMODEL_VERBOSE"):
    print(f"[dancingmodel] CUDA sources: {sources}")

  ext = torch_cpp_ext.CUDAExtension(
    name="dancingmodel._kernels",
    sources=sources,
    extra_compile_args=['-std=c++17'],
  )
  return [ext]


def _package_list():
  return find_packages(
    exclude=(
      "build",
      "include",
      "csrc",
      "test",
      "dist",
      "docs",
      "benchmarks",
      "dancingmodel.egg-info",
    )
  )


def main():
  # Only minimal arguments here; metadata & dependencies live in pyproject.toml
  kwargs = {}
  try:
    ext_modules = _build_ext_modules()
    if ext_modules:
      torch_cpp_ext = _try_import_torch_extension()
      kwargs.update({
        'ext_modules': ext_modules,
        'cmdclass': {'build_ext': torch_cpp_ext.BuildExtension},
      })
  except RuntimeError as e:
    # Gracefully degrade if extension build not critical
    print(f"[dancingmodel] Warning: {e}\nProceeding without CUDA extension.")

  setup(packages=_package_list(), **kwargs)


if __name__ == "__main__":
  main()
