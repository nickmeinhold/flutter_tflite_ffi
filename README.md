# flutter_tflite_ffi

A Flutter FFI plugin that provides direct bindings to the TensorFlow Lite C API for running machine learning inference on mobile devices.

## Features

- **Direct FFI Bindings**: Calls TensorFlow Lite C API directly via Dart FFI for optimal performance
- **Cross-Platform**: Supports both Android and iOS
- **Full Interpreter Control**: Load models, resize tensors, run inference, and retrieve results
- **Multiple Data Types**: Supports UInt8, Float32, Int32, and other tensor types
- **Image Utilities**: Built-in helper for loading and preprocessing images from files or asset bundles

## Installation

Add this to your `pubspec.yaml`:

```yaml
dependencies:
  flutter_tflite_ffi:
    git:
      url: https://github.com/nickmeinhold/flutter_tflite_ffi.git
```

## Usage

### Basic Example

```dart
import 'package:flutter_tflite_ffi/flutter_tflite_ffi.dart';

// Check TFLite version
print('TFLite version: ${version()}');

// Create an interpreter from a model file
final interpreter = createInterpreter(
  pathToModel: '/path/to/model.tflite',
  numThreads: 4,
);

// Inspect input/output tensors
print('Input tensors: ${interpreter.inputTensorCount}');
print('Output tensors: ${interpreter.outputTensorCount}');
print(interpreter.getInputTensorInfo());
print(interpreter.getOutputTensorInfo());

// Reshape input tensor if needed
interpreter.reshapeInputTensor(shape: [1, 224, 224, 3]);
interpreter.allocateTensors();

// Set input data and run inference
interpreter.setInputTensorData(rgbData.buffer);
interpreter.invoke();

// Get results
final output = interpreter.getOutputTensorData<double>();
print('Output: $output');

// Clean up
interpreter.delete();
```

### Loading Images

The plugin includes `EncodedFileImage` for loading and preprocessing images:

```dart
import 'package:flutter_tflite_ffi/flutter_tflite_ffi.dart';

// Load from asset bundle
final image = await EncodedFileImage.loadFromBundle(
  key: 'assets/test_image.jpg',
  inputImageFormat: ImageFormat.rgba8888,
  targetWidth: 224,
  targetHeight: 224,
);

// Or load from file
final image = await EncodedFileImage.loadFromFile(
  path: '/path/to/image.jpg',
  inputImageFormat: ImageFormat.rgba8888,
  targetWidth: 224,
  targetHeight: 224,
);

// Use with interpreter (for UInt8 models)
interpreter.setInputTensorData(image.rgbDataAsUint8s.buffer);

// Or for Float32 models
interpreter.setInputTensorData(image.rgbDataAsFloat32s.buffer);

// The image can also be displayed in Flutter
CustomPaint(painter: MyPainter(image.paintableImage));
```

## API Reference

### Top-Level Functions

| Function | Description |
|----------|-------------|
| `version()` | Returns the TensorFlow Lite version string |
| `createInterpreter({required String pathToModel, int? numThreads})` | Creates an interpreter from a `.tflite` model file |

### Interpreter

| Method | Description |
|--------|-------------|
| `inputTensorCount` | Number of input tensors |
| `outputTensorCount` | Number of output tensors |
| `getInputTensorInfo({int? index})` | Get info about an input tensor |
| `getOutputTensorInfo({int? index})` | Get info about an output tensor |
| `reshapeInputTensor({required List<int> shape, int? index})` | Resize an input tensor |
| `allocateTensors()` | Allocate memory for tensors (call after reshaping) |
| `setInputTensorData(ByteBuffer data, {int? index})` | Set input tensor data |
| `invoke()` | Run inference |
| `getOutputTensorData<T>({int? index})` | Get output tensor data (T: `double` or `int`) |
| `delete()` | Free interpreter resources |

### TensorInfo

Contains metadata about a tensor:

- `name`: Tensor name
- `shape`: Dimension sizes (e.g., `[1, 224, 224, 3]`)
- `bytes`: Total byte size
- `dataTypeName`: Type name (e.g., "Float32", "UInt8")
- `isVariable`: Whether the tensor is variable

## Supported Platforms

| Platform | Status |
|----------|--------|
| Android | Supported |
| iOS | Supported |
| macOS | Not yet |
| Windows | Not yet |
| Linux | Not yet |
| Web | Not supported (FFI) |

## Requirements

- Flutter SDK >= 3.0.0
- Dart SDK >= 3.0.0

## Development

### Regenerating FFI Bindings

If you need to update the TensorFlow Lite bindings:

```bash
flutter pub run ffigen --config ffigen.yaml
```

This requires the TensorFlow source headers to be available at `tensorflow/tensorflow/lite/c/`.

## Error Handling

The plugin throws `TFLiteStatusException` for TensorFlow Lite errors:

```dart
try {
  interpreter.invoke();
} on TFLiteStatusException catch (e) {
  print('TFLite error: $e');
}
```

## License

See LICENSE file for details.
