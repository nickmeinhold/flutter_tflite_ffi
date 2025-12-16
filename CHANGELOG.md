# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024

### Added

- Initial release
- TensorFlow Lite interpreter support via FFI
- Android (arm64-v8a) and iOS platform support
- Load models from file path
- Resize input tensors
- Set input tensor data (UInt8 and Float32)
- Run inference
- Get output tensor data (Float32 and Int32)
- Image loading utilities with `EncodedFileImage`
- Tensor info inspection
