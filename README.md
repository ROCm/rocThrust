# rocThrust

## The rocThrust repository is retired, please use the [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries) repository

Thrust is a parallel algorithm library. It has been ported to
[HIP](https://github.com/ROCm/HIP) and [ROCm](https://www.github.com/ROCm/ROCm), which use
the [rocPRIM](https://github.com/ROCm/rocPRIM) library. The HIP-ported library
works on HIP and ROCm software. Currently there is no CUDA backend in place.

## Requirements

Software requirements include:

* CMake (3.10.2 or later)
* AMD [ROCm](https://rocm.docs.amd.com) Software (1.8.0 or later)
  * Including the [HipCC](https://github.com/ROCm/HIP) compiler, which must be set
    as your C++ compiler for ROCm
* [rocPRIM](https://github.com/ROCm/rocPRIM) library
  * This is automatically downloaded and built by the CMake script
* Python 3.6 or higher (for HIP on Windows; only required for install scripts)
* Visual Studio 2019 with Clang support (for HIP on Windows)
* Strawberry Perl (for HIP on Windows)

Optional:

* [GoogleTest](https://github.com/google/googletest)
  * Required only for tests; building tests is enabled by default
  * This is automatically downloaded and built by the CMake script
* [doxygen](https://www.doxygen.nl/)
  * Required for building the documentation

For ROCm hardware requirements, refer to:

* [Linux support](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
* [Windows support](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html)

## Documentation

> [!NOTE]
> The published rocThrust documentation is available [here](https://rocm.docs.amd.com/projects/rocThrust/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).
