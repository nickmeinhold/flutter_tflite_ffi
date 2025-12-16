import 'dart:io' as io;
import 'dart:typed_data';
import 'dart:ui' as painting;

import 'package:flutter/services.dart';

/// The `camera` plugin has ImageFormatGroup with:
///    - `unknown`, `yuv420`, `bgra8888`, `jpeg`,
enum ImageFormat { rgba8888, bgra8888, yuv420 }

/// Should be described by the model spec and found in the model metadata
/// `rgbf32` : RGB frame, dtype = float32
/// `rgb888` : RGB frame, dtype =
enum InputTensorFormat { rgb888, rgbf32 }

/// An [EncodedFileImage] takes a path (for a file) or a key (for a bundle)
/// and
///  - decodes the corresponding image
///  - creates rgbData in ImageFormat.rgb888 format
///  - a [painting.Image] is created along the way and stored as
///    the [paintableImage] member, in order to be accessible for
///    displaying with Flutter.

/// The rgbData can be passed to an Interpreter.setInputTensorData to set a
/// tensor's data.
///
/// Create an [EncodedFileImage] with one of
///  - [EncodedFileImage.loadFromBundle]
///  - [EncodedFileImage.loadFromFile]
///
class EncodedFileImage {
  EncodedFileImage._(
    this.targetWidth,
    this.targetHeight,
    painting.Image image,
    Uint8List rgbDataAsUint8s,
    Float32List rgbDataAsFloat32s,
  )   : _rgbDataAsUint8s = rgbDataAsUint8s,
        _rgbDataAsFloat32s = rgbDataAsFloat32s,
        _paintableImage = image;

  final int targetWidth;
  final int targetHeight;
  final Uint8List _rgbDataAsUint8s;
  final Float32List _rgbDataAsFloat32s;
  final painting.Image _paintableImage;

  int get numPixels => targetWidth * targetHeight;
  Uint8List get rgbDataAsUint8s => _rgbDataAsUint8s;
  Float32List get rgbDataAsFloat32s => _rgbDataAsFloat32s;
  painting.Image get paintableImage => _paintableImage;

  static Future<EncodedFileImage> loadFromBundle({
    required String key,
    required ImageFormat inputImageFormat,
    required int targetWidth,
    required int targetHeight,
  }) async {
    // Extract the image file from the bundle
    final byteData = await rootBundle.load(key);
    final inputImageData = byteData.buffer.asUint8List();

    return _createImage(
      inputImageData: inputImageData,
      inputImageFormat: inputImageFormat,
      targetWidth: targetWidth,
      targetHeight: targetHeight,
    );
  }

  static Future<EncodedFileImage> loadFromFile({
    required String path,
    required ImageFormat inputImageFormat,
    required int targetWidth,
    required int targetHeight,
  }) async {
    // Read in the image file
    final inputImageData = io.File(path).readAsBytesSync();

    return _createImage(
      inputImageData: inputImageData,
      inputImageFormat: inputImageFormat,
      targetWidth: targetWidth,
      targetHeight: targetHeight,
    );
  }

  static Future<EncodedFileImage> _createImage({
    required Uint8List inputImageData,
    required ImageFormat inputImageFormat,
    required int targetWidth,
    required int targetHeight,
  }) async {
    final paintableImage = await _convertDataToPaintableImage(
      inputImageData,
      targetWidth,
      targetHeight,
    );

    final rgbDataAsUint8s = await _extractRgbDataAsUint8s(
      paintableImage,
      targetWidth * targetHeight,
    );
    final rgbDataAsFloat32s = await _extractRgbDataAsFloat32s(
      paintableImage,
      targetWidth * targetHeight,
    );

    return EncodedFileImage._(
      targetWidth,
      targetHeight,
      paintableImage,
      rgbDataAsUint8s,
      rgbDataAsFloat32s,
    );
  }

  static Future<painting.Image> _convertDataToPaintableImage(
    Uint8List imageData,
    int imageWidth,
    int imageHeight,
  ) async {
    final buffer = await painting.ImmutableBuffer.fromUint8List(imageData);
    final descriptor = await painting.ImageDescriptor.encoded(buffer);
    buffer.dispose();

    final codec = await descriptor.instantiateCodec(
      targetWidth: imageWidth,
      targetHeight: imageHeight,
    );
    final frameInfo = await codec.getNextFrame();

    return frameInfo.image;
  }

  static Future<Uint8List> _extractRgbDataAsUint8s(
    painting.Image paintableImage,
    int numPixels,
  ) async {
    final rgbaByteData = (await paintableImage.toByteData())!;

    final rgbBytes = Uint8List(numPixels * 3);
    for (var i = 0; i < numPixels; i++) {
      final rgbOffset = i * 3;
      final rgbaOffset = i * 4;
      rgbBytes[rgbOffset] = rgbaByteData.getUint8(rgbaOffset); // red
      rgbBytes[rgbOffset + 1] = rgbaByteData.getUint8(rgbaOffset + 1); // green
      rgbBytes[rgbOffset + 2] = rgbaByteData.getUint8(rgbaOffset + 2); // blue
    }

    return rgbBytes;
  }

  static Future<Float32List> _extractRgbDataAsFloat32s(
    painting.Image paintableImage,
    int numPixels,
  ) async {
    final rgbaByteData = (await paintableImage.toByteData())!;

    final rgbBytes = Float32List(numPixels * 3);
    for (var i = 0; i < numPixels; i++) {
      final rgbOffset = i * 3;
      final rgbaOffset = i * 4;
      rgbBytes[rgbOffset] = rgbaByteData.getUint8(rgbaOffset).toDouble(); // red
      rgbBytes[rgbOffset + 1] =
          rgbaByteData.getUint8(rgbaOffset + 1).toDouble(); // green
      rgbBytes[rgbOffset + 2] =
          rgbaByteData.getUint8(rgbaOffset + 2).toDouble(); // blue
    }

    return rgbBytes;
  }
}
