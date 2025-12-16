// ignore_for_file: lines_longer_than_80_chars, dangling_library_doc_comments

import 'package:flutter_tflite_ffi/src/bindings/flutter_tflite_ffi_bindings_generated.dart';

/// These come from TfLiteStatus enum in tensorflow/lite/core/c/c_api_types.h
/// We really need a way to know if the tensorflow library updates these,
/// I've created a Notion issue to track the problem for now.
///
/// There is a function "TfLiteTypeGetName" in the C API but we have mapped the
/// TfLiteStatus enum to a String that uses the relevant comment
/// from c_api_types.h to provide more info in the exception.

const Map<TfLiteStatus, String> _messageFrom = {
  TfLiteStatus.kTfLiteOk: 'Not an error: There is a bug in the plugin, please file an issue if you have time.',
  TfLiteStatus.kTfLiteError: 'kTfLiteError: Generally referring to an error in the runtime (i.e. interpreter).',
  TfLiteStatus.kTfLiteDelegateError: 'kTfLiteDelegateError: Generally referring to an error from a TfLiteDelegate itself.',
  TfLiteStatus.kTfLiteApplicationError: 'kTfLiteApplicationError: Generally referring to an error in applying a delegate due to '
      ' incompatibility between runtime and delegate, e.g., this error is returned '
      " when trying to apply a TF Lite delegate onto a model graph that's already "
      ' immutable.',
  TfLiteStatus.kTfLiteDelegateDataNotFound: 'kTfLiteDelegateDataNotFound: Generally referring to serialized delegate data not being found.\n'
      ' See tflite::delegates::Serialization.',
  TfLiteStatus.kTfLiteDelegateDataWriteError: 'kTfLiteDelegateDataWriteError: Generally referring to data-writing issues in delegate serialization.\n'
      'See tflite::delegates::Serialization.',
  TfLiteStatus.kTfLiteDelegateDataReadError: 'kTfLiteDelegateDataReadError: Generally referring to data-reading issues in delegate serialization.\n'
      'See tflite::delegates::Serialization.',
  TfLiteStatus.kTfLiteUnresolvedOps: 'kTfLiteUnresolvedOps: Generally referring to issues when the TF Lite model has ops that cannot be '
      'resolved at runtime. This could happen when the specific op is not '
      'registered or built with the TF Lite framework.',
  TfLiteStatus.kTfLiteCancelled: 'kTfLiteCancelled: Generally referring to invocation cancelled by the user.\n'
      'See `interpreter::Cancel`.\n',
};

class TFLiteStatusException implements Exception {
  TFLiteStatusException({required this.intro, required TfLiteStatus code})
      : tfLiteMessage = _messageFrom[code] ??
            'Unknown Error: TensorFlow Lite explicitly reserves the right to add new error '
                'status values in future in order to indicate more fine-grained internal states.\n'
                'If you are interested, please feel free to file an issue in the plugin repo and we '
                'can try and track down where the problem is';

  final String intro;
  final String tfLiteMessage;

  @override
  String toString() => '$intro\n$tfLiteMessage';
}
