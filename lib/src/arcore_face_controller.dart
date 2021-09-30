import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/services.dart';

import '../arcore_flutter_plugin.dart';
import 'package:vector_math/vector_math_64.dart';

typedef FacesEventHandler = void Function(String transform);

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

  FacesEventHandler onGetFacesNodes;

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
      case 'onGetFacesNodes':
        var matrixString = call.arguments.toString();
        onGetFacesNodes(matrixString);
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

  Future<dynamic> getFOV() {
    return _channel.invokeMethod('getFOV');
  }

  Future<List<Vector3>> getMeshVertices() async {
    final rawVertices =
        await (_channel.invokeMethod('getMeshVertices')) as List;

    List<Vector3> result = [];
    for (var i = 0; i < rawVertices.length - 1; i += 3) {
      result.add(Vector3(rawVertices[i] as double, rawVertices[i + 1] as double,
          rawVertices[i + 2] as double));
    }
    return result;
  }

  Future<List<int>> getMeshTriangleIndices() async {
    final rawIndices =
        await (_channel.invokeMethod('getMeshTriangleIndices')) as List;
    List<int> result = rawIndices.map((e) => e as int).toList();
    return result;
  }

  Future<dynamic> projectPoint(Vector3 point) async {
    final projectPoint = await _channel.invokeMethod('projectPoint', {
      'point': [point.x, point.y, point.z].toList()
    });
    return projectPoint;
  }

  void dispose() {
    _channel?.invokeMethod<void>('dispose');
  }
}
