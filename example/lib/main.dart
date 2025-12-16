import 'package:flutter/material.dart';
import 'package:flutter_tflite_ffi/flutter_tflite_ffi.dart';

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
      home: const TFLiteExample(),
    );
  }
}

class TFLiteExample extends StatefulWidget {
  const TFLiteExample({super.key});

  @override
  State<TFLiteExample> createState() => _TFLiteExampleState();
}

class _TFLiteExampleState extends State<TFLiteExample> {
  String _tfliteVersion = 'Unknown';
  String _status = 'Not initialized';

  @override
  void initState() {
    super.initState();
    _initTFLite();
  }

  void _initTFLite() {
    setState(() {
      _tfliteVersion = version();
      _status = 'TFLite initialized successfully';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('TFLite FFI Example'),
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.memory, size: 64),
              const SizedBox(height: 24),
              Text(
                'TensorFlow Lite Version',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 8),
              Text(
                _tfliteVersion,
                style: Theme.of(context).textTheme.headlineMedium,
              ),
              const SizedBox(height: 24),
              Text(
                'Status: $_status',
                style: Theme.of(context).textTheme.bodyLarge,
              ),
              const SizedBox(height: 48),
              const Text(
                'To run inference, add a .tflite model\n'
                'to the assets folder and use:\n\n'
                'final interpreter = createInterpreter(\n'
                '  pathToModel: modelPath,\n'
                ');',
                textAlign: TextAlign.center,
                style: TextStyle(fontFamily: 'monospace'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
