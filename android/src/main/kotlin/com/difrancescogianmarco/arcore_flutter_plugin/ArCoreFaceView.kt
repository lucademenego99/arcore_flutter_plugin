package com.difrancescogianmarco.arcore_flutter_plugin

import android.app.Activity
import android.content.Context
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import android.opengl.Matrix.*
import com.difrancescogianmarco.arcore_flutter_plugin.utils.ArCoreUtils
import com.google.ar.core.AugmentedFace
import com.google.ar.core.Config
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableException
import com.google.ar.sceneform.ArSceneView
import com.google.ar.sceneform.Scene
import com.google.ar.sceneform.rendering.ModelRenderable
import com.google.ar.sceneform.rendering.Renderable
import com.google.ar.sceneform.rendering.Texture
import com.google.ar.sceneform.ux.AugmentedFaceNode
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import kotlin.collections.HashMap
import kotlin.math.*

class ArCoreFaceView(activity:Activity,context: Context, messenger: BinaryMessenger, id: Int, debug: Boolean) : BaseArCoreView(activity, context, messenger, id, debug) {

    private val methodChannel2: MethodChannel = MethodChannel(messenger, "arcore_flutter_plugin_$id")
    private val TAG: String = ArCoreFaceView::class.java.name
    private var faceRegionsRenderable: ModelRenderable? = null
    private var faceMeshTexture: Texture? = null
    private val faceNodeMap = HashMap<AugmentedFace, AugmentedFaceNode>()
    private var faceSceneUpdateListener: Scene.OnUpdateListener

    init {
        faceSceneUpdateListener = Scene.OnUpdateListener { frameTime ->
            run {
                //                if (faceRegionsRenderable == null || faceMeshTexture == null) {
                // if (faceMeshTexture == null) {
                //     return@OnUpdateListener
                // }
                val faceList = arSceneView?.session?.getAllTrackables(AugmentedFace::class.java)

                faceList?.let {
                    // Make new AugmentedFaceNodes for any new faces.
                    for (face in faceList) {
                        if (!faceNodeMap.containsKey(face)) {
                            val faceNode = AugmentedFaceNode(face)
                            faceNode.setParent(arSceneView?.scene)
                            faceNode.faceRegionsRenderable = faceRegionsRenderable
                            faceNode.faceMeshTexture = faceMeshTexture
                            faceNodeMap[face] = faceNode

                            // change assets on runtime
                        } else if(faceNodeMap[face]?.faceRegionsRenderable != faceRegionsRenderable  ||  faceNodeMap[face]?.faceMeshTexture != faceMeshTexture ){
                            faceNodeMap[face]?.faceRegionsRenderable = faceRegionsRenderable
                            faceNodeMap[face]?.faceMeshTexture = faceMeshTexture
                        }
                    }

                    // Remove any AugmentedFaceNodes associated with an AugmentedFace that stopped tracking.
                    val iter = faceNodeMap.iterator()
                    while (iter.hasNext()) {
                        val entry = iter.next()
                        val face = entry.key
                        if (face.trackingState == TrackingState.STOPPED) {
                            val faceNode = entry.value
                            faceNode.setParent(null)
                            iter.remove()
                        }
                    }

                    val list = faceNodeMap.toList().map { it.first }
                    if (list.size > 0) {
                        val dest = FloatArray(16)
                        list[0].getCenterPose().toMatrix(dest, 0);
                        val doubleArray = DoubleArray(dest.size)
                        for ((i, a) in dest.withIndex()) {
                            doubleArray[i] = a.toDouble()
                        }
                        methodChannel2.invokeMethod("onGetFacesNodes", doubleArray)
                    }
                }
            }
        }
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        if(isSupportedDevice){
            debugLog(call.method +"called on supported device")
            when (call.method) {
                "init" -> {
                    arScenViewInit(call, result)
                }
                "loadMesh" -> {
                    val map = call.arguments as HashMap<*, *>
                    val textureBytes = map["textureBytes"] as ByteArray
                    val skin3DModelFilename = map["skin3DModelFilename"] as? String
                    loadMesh(textureBytes, skin3DModelFilename)
                }
                "getFOV" -> {
                    val dest = FloatArray(16)
                    arSceneView?.arFrame?.camera?.getProjectionMatrix(dest, 0, 0.0001f, 2.0f)
                    val res = 2 * atan(1/dest[5]) * 180/PI;
                    result.success(res)
                }
                "getMeshVertices" -> {
                    val list = faceNodeMap.toList().map { it.first }
                    if (list.size > 0) {
                        val vertices = list[0].getMeshVertices();
                        vertices.rewind();
                        val size = vertices.remaining();
                        val doubleArray = DoubleArray(size);
                        for (i in 0..size-1) {
                            doubleArray[i] = vertices.get().toDouble();
                        }
                        result.success(doubleArray);
                    }
                }
                "getMeshTriangleIndices" -> {
                    val list = faceNodeMap.toList().map { it.first }
                    if (list.size > 0) {
                        val vertices = list[0].getMeshTriangleIndices();
                        val size = vertices.remaining();
                        val intArray = IntArray(size)
                        for (i in 0..size-1) {
                            intArray[i] = vertices.get().toInt();
                        }
                        result.success(intArray)
                    }
                }
                "projectPoint" -> {
                    val map = call.arguments as HashMap<*, *>
                    val point = map["point"] as? ArrayList<Float>

                    if (point != null) {
                        val imageDimensions = arSceneView?.arFrame?.camera?.getImageIntrinsics()?.getImageDimensions()

                        if (imageDimensions != null) {
                            val projmtx = FloatArray(16)
                            arSceneView?.arFrame?.camera?.getProjectionMatrix(projmtx, 0, 0.0001f, 2.0f)

                            val viewmtx = FloatArray(16)
                            arSceneView?.arFrame?.camera?.getViewMatrix(viewmtx, 0)

                            val anchorMatrix = FloatArray(16)
                            setIdentityM(anchorMatrix, 0);
                            anchorMatrix[3] = point.get(0);
                            anchorMatrix[7] = point.get(1);
                            anchorMatrix[11] = point.get(2);

                            val worldToScreenMatrix = calculateWorldToCameraMatrix(anchorMatrix, viewmtx, projmtx);

                            val anchor_2d = worldToScreen(imageDimensions!![0], imageDimensions!![1], worldToScreenMatrix);

                            val doubleArrayTest = DoubleArray(4)
                            for ((i, a) in anchor_2d.withIndex()) {
                                doubleArrayTest[i] = a.toDouble()
                            }
                            result.success(doubleArrayTest);

                            // result.success(anchor_2d);
                        } else {
                            result.error("noImageDimensionsFound", "No camera found on arFrame", null);
                        }
                    } else {
                        result.error("noPointProvided", "The user didn't provide any point to project", null);
                    }
                }
                "dispose" -> {
                    debugLog( " updateMaterials")
                    dispose()
                }
                else -> {
                    result.notImplemented()
                }
            }
        }else{
            debugLog("Impossible call " + call.method + " method on unsupported device")
            result.error("Unsupported Device","",null)
        }
    }

    fun calculateWorldToCameraMatrix(modelmtx: FloatArray, viewmtx: FloatArray, prjmtx: FloatArray): FloatArray {
        val scaleFactor = 1.0f;
        val scaleMatrix = FloatArray(16)
        val modelXscale = FloatArray(16)
        val viewXmodelXscale = FloatArray(16)
        val worldToScreenMatrix = FloatArray(16)

        setIdentityM(scaleMatrix, 0);
        scaleMatrix[0] = scaleFactor;
        scaleMatrix[5] = scaleFactor;
        scaleMatrix[10] = scaleFactor;

        multiplyMM(modelXscale, 0, modelmtx, 0, scaleMatrix, 0);
        multiplyMM(viewXmodelXscale, 0, viewmtx, 0, modelXscale, 0);
        multiplyMM(worldToScreenMatrix, 0, prjmtx, 0, viewXmodelXscale, 0);

        return worldToScreenMatrix;
    }

    fun worldToScreen(screenWidth: Int, screenHeight: Int, worldToCameraMatrix: FloatArray): FloatArray {
        val origin = FloatArray(4)
        origin[0] = 0f;
        origin[1] = 0f;
        origin[2] = 0f;
        origin[3] = 1f;

        val ndcCoord = FloatArray(4)
        multiplyMV(ndcCoord, 0,  worldToCameraMatrix, 0,  origin, 0);

        ndcCoord[0] = (ndcCoord[0]/ndcCoord[3]).toFloat();
        ndcCoord[1] = (ndcCoord[1]/ndcCoord[3]).toFloat();

        return ndcCoord;

        //val pos_2d = DoubleArray(2)
        //pos_2d[0] = (screenWidth  * ((ndcCoord[0] + 1.0)/2.0));
        //pos_2d[1] = (screenHeight * (( 1.0 - ndcCoord[1])/2.0));

        //return pos_2d;
    }

    fun loadMesh(textureBytes: ByteArray?, skin3DModelFilename: String?) {
        if (skin3DModelFilename != null) {
            // Load the face regions renderable.
            // This is a skinned model that renders 3D objects mapped to the regions of the augmented face.
            ModelRenderable.builder()
                    .setSource(activity, Uri.parse(skin3DModelFilename))
                    .build()
                    .thenAccept { modelRenderable ->
                        faceRegionsRenderable = modelRenderable
                        modelRenderable.isShadowCaster = false
                        modelRenderable.isShadowReceiver = false
                    }
        }

        // Load the face mesh texture.
        Texture.builder()
                //.setSource(activity, Uri.parse("fox_face_mesh_texture.png"))
                .setSource(BitmapFactory.decodeByteArray(textureBytes, 0, textureBytes!!.size))
                .build()
                .thenAccept { texture -> faceMeshTexture = texture }
    }

    private fun arScenViewInit(call: MethodCall, result: MethodChannel.Result) {
        val enableAugmentedFaces: Boolean? = call.argument("enableAugmentedFaces")
        if (enableAugmentedFaces != null && enableAugmentedFaces) {
            // This is important to make sure that the camera stream renders first so that
            // the face mesh occlusion works correctly.
            arSceneView?.cameraStreamRenderPriority = Renderable.RENDER_PRIORITY_FIRST
            arSceneView?.scene?.addOnUpdateListener(faceSceneUpdateListener)
        }

        result.success(null)
    }

    override fun onResume() {
        if (arSceneView == null) {
            return
        }

        if (arSceneView?.session == null) {

            // request camera permission if not already requested
            if (!ArCoreUtils.hasCameraPermission(activity)) {
                ArCoreUtils.requestCameraPermission(activity, RC_PERMISSIONS)
            }

            // If the session wasn't created yet, don't resume rendering.
            // This can happen if ARCore needs to be updated or permissions are not granted yet.
            try {
                val session = ArCoreUtils.createArSession(activity, installRequested, true)
                if (session == null) {
                    installRequested = false
                    return
                } else {
                    val config = Config(session)
                    config.augmentedFaceMode = Config.AugmentedFaceMode.MESH3D
                    config.updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                    session.configure(config)
                    arSceneView?.setupSession(session)
                }
            } catch (e: UnavailableException) {
                ArCoreUtils.handleSessionException(activity, e)
            }
        }

        try {
            arSceneView?.resume()
        } catch (ex: CameraNotAvailableException) {
            ArCoreUtils.displayError(activity, "Unable to get camera", ex)
            activity.finish()
            return
        }

    }

    override fun onDestroy() {
        arSceneView?.scene?.removeOnUpdateListener(faceSceneUpdateListener)
        super.onDestroy()
    }

}
