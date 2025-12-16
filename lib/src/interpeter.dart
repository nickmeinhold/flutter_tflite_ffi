// ignore_for_file: lines_longer_than_80_chars

import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:flutter_tflite_ffi/flutter_tflite_ffi.dart';
import 'package:flutter_tflite_ffi/src/bindings/bindings_global.dart';
import 'package:flutter_tflite_ffi/src/bindings/flutter_tflite_ffi_bindings_generated.dart';
import 'package:flutter_tflite_ffi/src/exceptions/t_f_lite_status_exception.dart';
import 'package:flutter_tflite_ffi/src/utils/string_extensions.dart';

typedef TensorStruct = TfLiteTensor;

abstract class Interpreter {
  /// The total number of input tensors. 0 if the interpreter creation failed.
  int get inputTensorCount;

  /// The total number of output tensors. 0 if the interpreter creation failed.
  int get outputTensorCount;

  /// Returns information describing the input tensor at index.
  TensorInfo getInputTensorInfo({int? index});

  /// Returns information describing the output tensor at index.
  ///
  /// The shape and underlying data buffer for output tensors may be not be
  /// available until after the output tensor has been both sized and allocated.
  /// In general, best practice is to interact with the output tensor after
  /// calling [invoke()].
  TensorInfo getOutputTensorInfo({int? index});

  /// From the C function: TfLiteInterpreterResizeInputTensor
  ///   Resizes the specified input tensor.
  ///
  ///   NOTE: After a resize, the client must explicitly allocate tensors before
  ///   attempting to access the resized tensor data or invoke the interpreter.
  ///
  ///   REQUIRES: 0 <= input_index < TfLiteInterpreterGetInputTensorCount(tensor)
  ///
  ///   This function makes a copy of the input dimensions, so the client can
  ///   safely deallocate input_dims immediately after this function returns.
  void reshapeInputTensor({required List<int> shape, int? index});

  /// Updates allocations for all tensors, resizing dependent tensors using the
  /// specified input tensor dimensionality.
  ///
  /// This is a relatively expensive operation, and need only be called after
  /// creating the graph and/or resizing any inputs.
  void allocateTensors();

  void setInputTensorData(ByteBuffer rgbData, {int? index});

  /// Invokes the interpreter to run inference. May throw [TFLiteStatusException]
  /// if the underlying C function returned an error code.
  void invoke();

  /// Copy (?) the C memory into a Dart Uint8List
  /// Copies to the provided output buffer from the tensor's buffer.
  List<T> getOutputTensorData<T extends Object>({int? index});

  void delete();
}

class InterpreterOptions {
  InterpreterOptions({this.numThreads = 1});

  int numThreads;
}

class NativeInterpreter implements Interpreter {
  NativeInterpreter(String modelPath, {int? numThreads}) {
    final pathPtr = modelPath.toCStringWithAllocate();
    _modelPtr = bindingsGlobal.TfLiteModelCreateFromFile(pathPtr);
    malloc.free(pathPtr);

    /// Setup the options used when creating the Interpreter
    final optionsPtr = bindingsGlobal.TfLiteInterpreterOptionsCreate();
    bindingsGlobal.TfLiteInterpreterOptionsSetNumThreads(
      optionsPtr,
      numThreads ?? 1,
    );

    // Create the interpreter
    _ptr = bindingsGlobal.TfLiteInterpreterCreate(_modelPtr, optionsPtr);

    // Delete the options as they are no longer needed
    bindingsGlobal.TfLiteInterpreterOptionsDelete(optionsPtr);
  }

  late Pointer<TfLiteInterpreter> _ptr;
  late Pointer<TfLiteModel> _modelPtr;

  @override
  int get inputTensorCount =>
      bindingsGlobal.TfLiteInterpreterGetInputTensorCount(_ptr);
  @override
  int get outputTensorCount =>
      bindingsGlobal.TfLiteInterpreterGetOutputTensorCount(_ptr);

  @override
  TensorInfo getInputTensorInfo({int? index}) =>
      bindingsGlobal.TfLiteInterpreterGetInputTensor(_ptr, index ?? 0).toInfo();
  @override
  TensorInfo getOutputTensorInfo({int? index}) =>
      bindingsGlobal.TfLiteInterpreterGetOutputTensor(_ptr, index ?? 0)
          .toInfo();

  @override
  void reshapeInputTensor({required List<int> shape, int? index}) {
    // Load up some C memory
    final inputDims = malloc<Int>(shape.length);
    for (var index = 0; index < shape.length; index++) {
      inputDims[index] = shape[index];
    }

    // Pass the C memory into the  relevant C function
    final result = bindingsGlobal.TfLiteInterpreterResizeInputTensor(
      _ptr,
      index ?? 0,
      inputDims,
      shape.length,
    );
    if (result != TfLiteStatus.kTfLiteOk) {
      throw TFLiteStatusException(
        intro: 'Reshaping a tensor gave:',
        code: result,
      );
    }

    // Free the C memory!
    malloc.free(inputDims);
  }

  @override
  void allocateTensors() {
    final result = bindingsGlobal.TfLiteInterpreterAllocateTensors(_ptr);
    if (result != TfLiteStatus.kTfLiteOk) {
      throw TFLiteStatusException(
        intro: 'Allocating tensors gave: ',
        code: result,
      );
    }
  }

  @override
  void setInputTensorData(ByteBuffer rgbData, {int? index}) {
    /// We could do this:
    /// bindingsGlobal.TfLiteTensorCopyFromBuffer(inputTensorPtr, ?, data.length);
    ///
    /// But then we need to...
    /// - allocate C memory
    /// - copy rgb bytes into C memory
    /// - pass pointer to C memory into TfLiteTensorCopyFromBuffer
    ///
    /// The TfLiteTensorCopyFromBuffer function calls the following C++:
    ///
    /// if (tensor->bytes != input_data_size) {
    ///   return kTfLiteError;
    /// }
    /// memcpy(tensor->data.raw, input_data, input_data_size);
    /// return kTfLiteOk;
    ///
    /// So rather than allocating C memory, copying into it, then calling
    /// TfLiteTensorCopyFromBuffer (which does another memcpy) we can copy straight
    /// into the tensor.
    ///
    /// We might be able to improve on this further when TypedData unwrapping is done (#44589)

    final inputTensorPtr =
        bindingsGlobal.TfLiteInterpreterGetInputTensor(_ptr, index ?? 0);

    final tensor = inputTensorPtr.ref;

    if (tensor.bytes != rgbData.lengthInBytes) {
      throw TFLiteStatusException(
        intro: 'When setting input tensor data, the passed rgb data '
            '(${rgbData.lengthInBytes} bytes) was not the same size as the '
            'allocated tensor data (${tensor.bytes} bytes), which threw:',
        code: TfLiteStatus.kTfLiteError,
      );
    }

    if (tensor.type == TfLiteType.kTfLiteUInt8.value) {
      final buf = tensor.data.raw.cast<Uint8>();
      final castList = rgbData.asUint8List();
      final numUint8s = rgbData.lengthInBytes;
      for (var i = 0; i < numUint8s; i++) {
        buf[i] = castList[i];
      }
    } else if (tensor.type == TfLiteType.kTfLiteFloat32.value) {
      final buf = tensor.data.raw.cast<Float>();
      final castList = rgbData.asFloat32List();
      final numFloat32s = rgbData.lengthInBytes ~/ 4;
      for (var i = 0; i < numFloat32s; i++) {
        buf[i] = castList[i];
      }
    }
  }

  @override
  void invoke() {
    final result = bindingsGlobal.TfLiteInterpreterInvoke(_ptr);
    if (result != TfLiteStatus.kTfLiteOk) {
      throw TFLiteStatusException(
        intro: 'When invoking the interpreter:',
        code: result,
      );
    }
  }

  @override
  List<T> getOutputTensorData<T extends Object>({int? index}) {
    final outputTensor =
        bindingsGlobal.TfLiteInterpreterGetOutputTensor(_ptr, index ?? 0);

    final tensorSizeInBytes = outputTensor.ref.bytes;

    final buffer = malloc.allocate<Void>(tensorSizeInBytes);

    final result = bindingsGlobal.TfLiteTensorCopyToBuffer(
      outputTensor,
      buffer,
      tensorSizeInBytes,
    );
    if (result != TfLiteStatus.kTfLiteOk) {
      throw TFLiteStatusException(
        intro: 'When getting output tensor data:',
        code: result,
      );
    }

    final List<dynamic> outputData;
    final tensorType = outputTensor.ref.type;
    if (T == double) {
      if (tensorType == TfLiteType.kTfLiteFloat32.value) {
        final castBuffer = buffer.cast<Float>();
        final numFloat32s = tensorSizeInBytes ~/ 4;
        outputData = Float32List(numFloat32s);

        for (var i = 0; i < numFloat32s; i++) {
          outputData[i] = castBuffer[i];
        }
      } else {
        throw Exception(
          'outputTensor.ref.type $tensorType was not recognized for double output.',
        );
      }
    } else if (T == int) {
      if (tensorType == TfLiteType.kTfLiteInt32.value) {
        final castBuffer = buffer.cast<Int32>();
        final numInt32s = tensorSizeInBytes ~/ 4;
        outputData = Int32List(numInt32s);

        for (var i = 0; i < numInt32s; i++) {
          outputData[i] = castBuffer[i];
        }
      } else if (tensorType == TfLiteType.kTfLiteUInt8.value) {
        final castBuffer = buffer.cast<Uint8>();
        outputData = Uint8List(tensorSizeInBytes);

        for (var i = 0; i < tensorSizeInBytes; i++) {
          outputData[i] = castBuffer[i];
        }
      } else if (tensorType == TfLiteType.kTfLiteInt8.value) {
        final castBuffer = buffer.cast<Int8>();
        outputData = Int8List(tensorSizeInBytes);

        for (var i = 0; i < tensorSizeInBytes; i++) {
          outputData[i] = castBuffer[i];
        }
      } else {
        throw Exception(
          'outputTensor.ref.type $tensorType was not recognized for int output.',
        );
      }
    } else {
      throw Exception(
        'You called getOutputTensorData with in invalid type parameter.',
      );
    }

    malloc.free(buffer);

    return outputData as List<T>;
  }

  @override
  void delete() {
    bindingsGlobal
      ..TfLiteModelDelete(_modelPtr)
      ..TfLiteInterpreterDelete(_ptr);
  }
}
