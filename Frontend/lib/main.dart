import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'dart:html' as html;
import 'dart:async';
import 'dart:convert';
import 'dart:ui' as ui;
import 'dart:typed_data';

void main() {
  runApp(const MaterialApp(
    title: 'Camera Web App',
    home: CameraWebApp(),
  ));
}

class CameraWebApp extends StatefulWidget {
  const CameraWebApp({Key? key}) : super(key: key);

  @override
  _CameraWebAppState createState() => _CameraWebAppState();
}

class _CameraWebAppState extends State<CameraWebApp> {
  late html.VideoElement _videoElement;
  bool _cameraInitialized = false;
  bool _isTakingPicture = false;
  String? _imagePath;
  final FlutterTts flutterTts = FlutterTts();
  String _viewId = 'video-${DateTime.now().millisecondsSinceEpoch}';

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    _videoElement = html.VideoElement()
      ..autoplay = true
      ..muted = true
      ..style.objectFit = 'cover'
      ..style.width = '100%'
      ..style.height = '100%';

    ui.platformViewRegistry.registerViewFactory(_viewId, (int viewId) {
      return _videoElement;
    });

    try {
      final mediaStream = await html.window.navigator.mediaDevices?.getUserMedia({
        'video': {
          'facingMode': 'environment',
          'width': { 'ideal': 3000 },
          'height': { 'ideal': 4000 },
        },
        'audio': false,
      });

      _videoElement.srcObject = mediaStream;
      setState(() {
        _cameraInitialized = true;
      });
    } catch (e) {
      print('Error initializing camera: $e');
      setState(() {
        _cameraInitialized = false;
      });
    }
  }

  Future<void> _takePicture() async {
    if (!_cameraInitialized || _isTakingPicture) return;

    setState(() {
      _isTakingPicture = true;
    });

    try {
      // Create a high-resolution canvas
      final html.CanvasElement canvas = html.CanvasElement(
        width: 1500, // Resize width
        height: 2000, // Resize height
      );

      final ctx = canvas.context2D;
      ctx.drawImageScaled(_videoElement, 0, 0, 1500, 2000);

      final imageDataUrl = canvas.toDataUrl('image/png');

      setState(() {
        _imagePath = imageDataUrl;
      });

      await _sendToAPI(imageDataUrl);
    } catch (e) {
      print('Error taking picture: $e');
    }

    setState(() {
      _isTakingPicture = false;
    });
  }

  Future<void> _sendToAPI(String imageDataUrl) async {
    try {
      const apiUrl = 'https://aiclub.uit.edu.vn/gpu150/ohmni_be/map_depth';
      final String base64Image = imageDataUrl.split(',')[1];
      final List<int> imageBytes = base64Decode(base64Image);
      
      final formData = html.FormData();
      final blob = html.Blob([Uint8List.fromList(imageBytes)], 'image/png');
      
      formData.appendBlob('file', blob, 'image.png');

      final request = html.HttpRequest();
      request.open('POST', apiUrl);
      request.onLoadEnd.listen((event) {
        if (request.status == 200) {
          if (request.responseText != null) {
            Map<String, dynamic> responseJson = jsonDecode(request.responseText!);
            String text = responseJson['output'] ?? "No one";
            print('infor: ${text}');
            _speak(text);
          } else {
            print('Error: responseText is null');
          }
        } else {
          print('Lỗi upload: ${request.statusText}');
          _speak('Sorry');
        }
        _resetCamera();
      });

      request.send(formData);
    } catch (e) {
      print('Lỗi gửi ảnh: $e');
    }
  }

  Future<void> _speak(String text) async {
    await flutterTts.setLanguage("vi-VN");
    await flutterTts.setPitch(1.0);
    await flutterTts.speak(text);
  }

  void _resetCamera() {
    setState(() {
      _imagePath = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Camera Web App'),
        backgroundColor: Colors.black,
      ),
      body: _buildCameraScreen(),
    );
  }

  Widget _buildCameraScreen() {
    return Column(
      children: [
        Expanded(
          child: _cameraInitialized
              ? HtmlElementView(viewType: _viewId)
              : const Center(
                  child: CircularProgressIndicator(),
                ),
        ),
        Container(
          height: 100,
          width: double.infinity,
          color: Colors.black,
          child: Center(
            child: _isTakingPicture
                ? const CircularProgressIndicator()
                : FloatingActionButton(
                    onPressed: _takePicture,
                    backgroundColor: Colors.white,
                    child: const Icon(Icons.camera_alt, color: Colors.black),
                  ),
          ),
        ),
      ],
    );
  }

  @override
  void dispose() {
    try {
      if (_videoElement.srcObject != null) {
        _videoElement.srcObject!.getTracks().forEach((track) => track.stop());
      }
    } catch (e) {
      print('Error stopping video tracks: $e');
    }
    super.dispose();
  }
}
