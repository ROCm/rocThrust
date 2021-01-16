# Change Log for rocThrust

Full documentation for rocThrust is available at [https://rocthrust.readthedocs.io/en/latest/](https://rocthrust.readthedocs.io/en/latest/)

## [Unreleased rocThrust-2.10.8 for ROCm 4.1.0]
### Added
- Updated to upstream Thrust 1.10.0

## [rocThrust-2.10.7 for ROCm 4.0.0]
### Added
- Implemented runtime error for unsupported algorithms and disabled respective tests.
- Updated CMake to use downloaded rocPRIM.

## [rocThrust-2.10.6 for ROCm 3.10]
### Added
- Added copy_if on device test case
### Known issues
- ROCm support for device malloc has been disabled. As a result, rocThrust functionality dependent on device malloc does not work. Please avoid using device launched thrust::sort and thrust::sort_by_key. Host launched functionality is not impacted. A partial enablement of device malloc is possible by setting HIP_ENABLE_DEVICE_MALLOC to 1. Thrust::sort and thrust::sort_by_key may work on certain input sizes but is not recommended for production code.

## [rocThrust-2.10.5 for ROCm 3.9.0]
### Added
- Updated to upstream Thrust 1.9.8
- New test cases for device-side algorithms
### Fixes
- Bugfix for binary search
- Implemented workarounds for hipStreamDefault hang
### Known issues
- ROCm support for device malloc has been disabled. As a result, rocThrust functionality dependent on device malloc does not work. Please avoid using device launched thrust::sort and thrust::sort_by_key. Host launched functionality is not impacted. A partial enablement of device malloc is possible by setting HIP_ENABLE_DEVICE_MALLOC to 1. Thrust::sort and thrust::sort_by_key may work on certain input sizes but is not recommended for production code.

## [rocThrust-2.10.4 for ROCm 3.8.0]
### Added
- No new features
### Known issues
- ROCm support for device malloc has been disabled. As a result, rocThrust functionality dependent on device malloc does not work. Please avoid using device launched thrust::sort and thrust::sort_by_key. Host launched functionality is not impacted. A partial enablement of device malloc is possible by setting HIP_ENABLE_DEVICE_MALLOC to 1. Thrust::sort and thrust::sort_by_key may work on certain input sizes but is not recommended for production code.

## [rocThrust-2.10.3 for ROCm 3.7.0]
### Added
- Updated to upstream Thrust 1.9.4
### Changed
- Package dependecy change to rocprim only
### Known issues
- ROCm support for device malloc has been disabled. As a result, rocThrust functionality dependent on device malloc does not work. Please avoid using device launched thrust::sort and thrust::sort_by_key. Host launched functionality is not impacted. A partial enablement of device malloc is possible by setting HIP_ENABLE_DEVICE_MALLOC to 1. Thrust::sort and thrust::sort_by_key may work on certain input sizes but is not recommended for production code.

## [rocThrust-2.10.2 for ROCm 3.6.0]
### Added
- No new features
### Known Issues
- ROCm support for device malloc has been disabled. As a result, rocThrust functionality dependent on device malloc does not work. Please avoid using device launched thrust::sort and thrust::sort_by_key. Host launched functionality is not impacted. A partial enablement of device malloc is possible by setting HIP_ENABLE_DEVICE_MALLOC to 1. Thrust::sort and thrust::sort_by_key may work on certain input sizes but is not recommended for production code.

## [rocThrust-2.10.1 for ROCm 3.5.0]
### Added
- Improved tests with fixed and random seeds for test data
### Changed
- CMake searches for rocThrust locally first; downloads from github if local search fails
### Deprecated
- HCC build deprecated
