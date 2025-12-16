import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tflite_ffi/flutter_tflite_ffi.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'TFLite FFI Example',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const ImageClassificationDemo(),
    );
  }
}

class ImageClassificationDemo extends StatefulWidget {
  const ImageClassificationDemo({super.key});

  @override
  State<ImageClassificationDemo> createState() =>
      _ImageClassificationDemoState();
}

class _ImageClassificationDemoState extends State<ImageClassificationDemo> {
  String _tfliteVersion = 'Unknown';
  String _status = 'Initializing...';
  List<String> _labels = [];
  List<MapEntry<String, int>> _results = [];
  bool _isLoading = true;
  Interpreter? _interpreter;

  @override
  void initState() {
    super.initState();
    _initTFLite();
  }

  @override
  void dispose() {
    _interpreter?.delete();
    super.dispose();
  }

  Future<void> _initTFLite() async {
    try {
      setState(() {
        _tfliteVersion = version();
        _status = 'Loading model...';
      });

      // Load labels
      final labelsData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelsData.split('\n');

      // Copy model from assets to a file (TFLite needs a file path)
      final modelPath = await _copyAssetToFile('assets/mobilenet_v1_quant.tflite');

      // Create interpreter
      _interpreter = createInterpreter(pathToModel: modelPath, numThreads: 4);

      setState(() {
        _status = 'Model loaded. Tap "Classify" to run inference.';
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _status = 'Error: $e';
        _isLoading = false;
      });
    }
  }

  Future<String> _copyAssetToFile(String assetPath) async {
    final byteData = await rootBundle.load(assetPath);
    final tempDir = await getTemporaryDirectory();
    final fileName = assetPath.split('/').last;
    final file = File('${tempDir.path}/$fileName');
    await file.writeAsBytes(byteData.buffer.asUint8List());
    return file.path;
  }

  Future<void> _classifyImage() async {
    if (_interpreter == null) return;

    setState(() {
      _status = 'Running inference...';
      _isLoading = true;
    });

    try {
      // Load and preprocess the test image
      final image = await EncodedFileImage.loadFromBundle(
        key: 'assets/test_image.jpg',
        inputImageFormat: ImageFormat.rgba8888,
        targetWidth: 224,
        targetHeight: 224,
      );

      // Get input tensor info
      final inputInfo = _interpreter!.getInputTensorInfo();

      // Allocate tensors
      _interpreter!.allocateTensors();

      // Set input data (quantized model expects UInt8)
      _interpreter!.setInputTensorData(image.rgbDataAsUint8s.buffer);

      // Run inference
      _interpreter!.invoke();

      // Get output (quantized model outputs UInt8, but we read as int)
      final outputInfo = _interpreter!.getOutputTensorInfo();

      // For quantized models, output is typically UInt8 values 0-255
      // representing probabilities
      final outputTensor = _interpreter!.getOutputTensorData<int>();

      // Find top 5 results
      final indexed = outputTensor.asMap().entries.toList()
        ..sort((a, b) => b.value.compareTo(a.value));

      final top5 = indexed.take(5).map((e) {
        final label = e.key < _labels.length ? _labels[e.key] : 'Unknown';
        return MapEntry(label, e.value);
      }).toList();

      setState(() {
        _results = top5;
        _status = 'Classification complete!\n'
            'Input: ${inputInfo.shape} ${inputInfo.dataTypeName}\n'
            'Output: ${outputInfo.shape} ${outputInfo.dataTypeName}';
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _status = 'Error during inference: $e';
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('MobileNet Classification'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // TFLite version
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    const Icon(Icons.memory, size: 32),
                    const SizedBox(height: 8),
                    Text(
                      'TensorFlow Lite v$_tfliteVersion',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),

            // Test image
            Card(
              clipBehavior: Clip.antiAlias,
              child: Column(
                children: [
                  Image.asset(
                    'assets/test_image.jpg',
                    height: 200,
                    width: double.infinity,
                    fit: BoxFit.cover,
                  ),
                  Padding(
                    padding: const EdgeInsets.all(8),
                    child: Text(
                      'Test Image (will be resized to 224x224)',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // Classify button
            FilledButton.icon(
              onPressed: _isLoading ? null : _classifyImage,
              icon: _isLoading
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.play_arrow),
              label: Text(_isLoading ? 'Processing...' : 'Classify Image'),
            ),
            const SizedBox(height: 16),

            // Status
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Text(
                  _status,
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ),
            ),
            const SizedBox(height: 16),

            // Results
            if (_results.isNotEmpty) ...[
              Text(
                'Top 5 Predictions:',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 8),
              ...List.generate(_results.length, (index) {
                final result = _results[index];
                final confidence = result.value / 255.0; // Normalize UInt8
                return Card(
                  child: ListTile(
                    leading: CircleAvatar(
                      child: Text('${index + 1}'),
                    ),
                    title: Text(result.key),
                    subtitle: LinearProgressIndicator(
                      value: confidence,
                    ),
                    trailing: Text('${(confidence * 100).toStringAsFixed(1)}%'),
                  ),
                );
              }),
            ],
          ],
        ),
      ),
    );
  }
}
