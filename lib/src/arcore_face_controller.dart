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

  List<double> vector3ToJson(Vector3 point) {
    final list = List.filled(3, 0.0);
    point.copyIntoArray(list);
    return list;
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

  Future<dynamic> getMeshVertices() async {
    return await (_channel.invokeListMethod<double>('getMeshVertices')
        as FutureOr<List<dynamic>>);
  }

  Future<dynamic> getMeshTriangleIndices() async {
    return await (_channel.invokeListMethod<int>('getMeshTriangleIndices'))
        as FutureOr<List<dynamic>>;
  }

  Future<dynamic> projectPoint(Vector3 point) async {
    final projectPoint = await _channel.invokeListMethod<double>(
        'projectPoint', {'point': vector3ToJson(point)});
    return projectPoint;
  }

  void dispose() {
    _channel?.invokeMethod<void>('dispose');
  }
}
