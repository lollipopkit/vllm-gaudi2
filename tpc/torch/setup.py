# SPDX-License-Identifier: Apache-2.0
# Builds the gdn_tpc, gaudi2_fp8_tpc, and inkling2_tpc torch extensions: python setup.py build_ext --inplace
from pathlib import Path

import pybind11
from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir
from setuptools import setup
from torch.utils import cpp_extension

_TPC_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")


def _ext(name, src):
    return cpp_extension.CppExtension(
        name,
        [src],
        extra_compile_args=["-std=c++17"],
        libraries=["habana_pytorch2_plugin"],
        library_dirs=[get_lib_dir()],
        include_dirs=[get_include_dir(), "/usr/include/habanalabs", pybind11.get_include(), _TPC_SRC_DIR],
    )


setup(
    name="gaudi2_tpc_ops",
    ext_modules=[
        _ext("gdn_tpc", "gdn_tpc_op.cpp"),
        _ext("gaudi2_fp8_tpc", "gaudi2_fp8_tpc_op.cpp"),
        _ext("inkling2_tpc", "inkling2_tpc_op.cpp"),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
