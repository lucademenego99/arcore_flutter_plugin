import 'dart:typed_data';
import 'package:vector_math/vector_math_64.dart';

import 'package:flutter/cupertino.dart';
import 'package:flutter/services.dart';

import '../arcore_flutter_plugin.dart';

class ArCoreFaceController {
  ArCoreFaceController(
      {int id, this.enableAugmentedFaces, this.debug = false}) {
    _channel = MethodChannel('arcore_flutter_plugin_$id');
    _channel.setMethodCallHandler(_handleMethodCalls);
    init();
  }

  final bool enableAugmentedFaces;
  final bool debug;
  MethodChannel _channel;
  StringResultHandler onError;
  Vector3 _facesNodes;

  String get facesNodes {
    return _facesNodes;
  }

  init() async {
    try {
      await _channel.invokeMethod<void>('init', {
        'enableAugmentedFaces': enableAugmentedFaces,
      });
    } on PlatformException catch (ex) {
      print(ex.message);
    }
  }

  Future<dynamic> _handleMethodCalls(MethodCall call) async {
    if (debug) {
      print('_platformCallHandler call ${call.method} ${call.arguments}');
    }
    switch (call.method) {
      case 'onError':
        if (onError != null) {
          onError(call.arguments);
        }
        break;
      case 'getFacesNodes':
        _facesNodes = call.arguments;
        print(call.arguments);
        break;
      default:
        if (debug) {
          print('Unknown method ${call.method}');
        }
    }
    return Future.value();
  }

  Future<void> loadMesh(
      {@required Uint8List textureBytes, String skin3DModelFilename}) {
    assert(textureBytes != null);
    return _channel.invokeMethod('loadMesh', {
      'textureBytes': textureBytes,
      'skin3DModelFilename': skin3DModelFilename
    });
  }
  
  Future<void> getFacesNodes() {
    return _channel.invokeMethod('getFacesNodes');
  }

  void dispose() {
    _channel?.invokeMethod<void>('dispose');
  }
}
