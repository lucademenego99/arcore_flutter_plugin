package com.difrancescogianmarco.arcore_flutter_plugin

import android.app.Activity
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.SurfaceTexture
import android.hardware.camera2.*
import android.hardware.camera2.CameraCaptureSession.CaptureCallback
import android.media.Image
import android.media.ImageReader
import android.net.Uri
import android.opengl.Matrix.*
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import android.view.*
import androidx.annotation.NonNull
import com.difrancescogianmarco.arcore_flutter_plugin.utils.ArCoreUtils
import com.google.ar.core.AugmentedFace
import com.google.ar.core.Config
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableException
import com.google.ar.sceneform.Scene
import com.google.ar.sceneform.rendering.ModelRenderable
import com.google.ar.sceneform.rendering.Renderable
import com.google.ar.sceneform.rendering.Texture
import com.google.ar.sceneform.ux.AugmentedFaceNode
import com.google.mediapipe.components.ExternalTextureConverter
import com.google.mediapipe.components.FrameProcessor
import com.google.mediapipe.formats.proto.LandmarkProto
import com.google.mediapipe.framework.AndroidAssetUtil
import com.google.mediapipe.framework.Packet
import com.google.mediapipe.framework.PacketGetter
import com.google.mediapipe.glutil.EglManager
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.*
import javax.microedition.khronos.egl.EGL10
import javax.microedition.khronos.egl.EGLContext
import kotlin.collections.HashMap
import kotlin.math.PI
import kotlin.math.atan

class ArCoreFaceView(activity:Activity,context: Context, messenger: BinaryMessenger, id: Int, debug: Boolean) : BaseArCoreView(activity, context, messenger, id, debug), ImageReader.OnImageAvailableListener, SurfaceTexture.OnFrameAvailableListener {
    private val methodChannel2: MethodChannel = MethodChannel(messenger, "arcore_flutter_plugin_$id")
    private val TAG: String = ArCoreFaceView::class.java.name
    private var faceRegionsRenderable: ModelRenderable? = null
    private var faceMeshTexture: Texture? = null
    private val faceNodeMap = HashMap<AugmentedFace, AugmentedFaceNode>()
    private var faceSceneUpdateListener: Scene.OnUpdateListener

    private var eglManager: EglManager? = null
    private var processor: FrameProcessor? = null
    private var converter: ExternalTextureConverter? = null
    private var previewDisplayView: SurfaceView? = null
    private var previewFrameTexture: SurfaceTexture? = null

    // Looper handler thread.
    private var backgroundThread: HandlerThread? = null

    // Looper handler.
    private var backgroundHandler: Handler? = null

    // Camera device. Used by both non-AR and AR modes.
    private var cameraDevice: CameraDevice? = null

    // Camera capture session. Used by both non-AR and AR modes.
    private var captureSession: CameraCaptureSession? = null

    // Image reader that continuously processes CPU images.
    private val cpuImageReader: ImageReader? = null

    // Camera preview capture request builder
    private var previewCaptureRequestBuilder: CaptureRequest.Builder? = null

    // Whether the app is currently in AR mode. Initial value determines initial state.
    private val arMode = true

    // Whether ARCore is currently active.
    private var arcoreActive = true

    private val FOCAL_LENGTH_STREAM_NAME = "focal_length_pixel"
    private val OUTPUT_LANDMARKS_STREAM_NAME = "face_landmarks_with_iris"

    companion object {
        init {
            // Load all native libraries needed by the app.
            System.loadLibrary("mediapipe_jni")
            System.loadLibrary("opencv_java3")
        }
    }

    init {
        AndroidAssetUtil.initializeNativeAssetManager(context);

        eglManager = EglManager((EGLContext.getEGL() as (EGL10)).eglGetCurrentContext())
        processor = FrameProcessor(activity, eglManager!!.nativeContext, "iris_tracking_gpu.binarypb", "input_video","output_video")
        processor!!.videoSurfaceOutput.setFlipY(true)
        converter = ExternalTextureConverter(eglManager!!.context)
        converter!!.setFlipY(true)
        converter!!.setConsumer(processor!!)

        previewDisplayView = SurfaceView(activity)

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
                        list[0].centerPose.toMatrix(dest, 0);
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

    private fun getLandmarksDebugString(landmarks: LandmarkProto.NormalizedLandmarkList): String? {
        var landmarkIndex = 0
        var landmarksString = ""
        for (landmark in landmarks.landmarkList) {
            landmarksString += """[$landmarkIndex]: (${landmark.x}, ${landmark.y}, ${landmark.z})"""
            ++landmarkIndex
        }
        return landmarksString
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
                    val width = map["width"] as? Int
                    val height = map["height"] as? Int

                    if (point != null) {
                        if (width != null && height != null) {
                            val projmtx = FloatArray(16)
                            arSceneView?.arFrame?.camera?.getProjectionMatrix(projmtx, 0, 0.0001f, 2.0f)

                            val viewmtx = FloatArray(16)
                            arSceneView?.arFrame?.camera?.getViewMatrix(viewmtx, 0)

                            val anchorMatrix = FloatArray(16)
                            setIdentityM(anchorMatrix, 0);
                            anchorMatrix[12] = point.get(0);
                            anchorMatrix[13] = point.get(1);
                            anchorMatrix[14] = point.get(2);

                            val worldToScreenMatrix = calculateWorldToCameraMatrix(anchorMatrix, viewmtx, projmtx);

                            val anchor_2d = worldToScreen(width, height, worldToScreenMatrix);

                            result.success(anchor_2d);
                        } else {
                            result.error("noImageDimensionsFound", "The user didn't provide image dimensions", null);
                        }
                    } else {
                        result.error("noPointProvided", "The user didn't provide any point to project", null);
                    }
                }
                "takeScreenshot" -> {
                    takeScreenshot(call, result);
                }
                "enableIrisTracking" -> {

                    // Store the ID of the camera used by ARCore.
                    val cameraId = arSceneView?.session?.cameraConfig?.cameraId;
                    // Use the currently configured CPU image size.

                    // Use the currently configured CPU image size.
                    val desiredCpuImageSize: Size = arSceneView!!.session!!.cameraConfig!!.imageSize
                    var cpuImageReader =
                            ImageReader.newInstance(
                                    desiredCpuImageSize.width,
                                    desiredCpuImageSize.height,
                                    ImageFormat.YUV_420_888,
                                    2);
                    cpuImageReader.setOnImageAvailableListener(this, backgroundHandler);
                    // When ARCore is running, make sure it also updates our CPU image surface.
                    arSceneView?.session?.sharedCamera?.setAppSurfaces(cameraId, listOf(cpuImageReader.surface));

                    pauseARCore()
                    resumeCamera2()

                    /*var cameraHelper = CameraXPreviewHelper()
                    cameraHelper.setOnCameraStartedListener { surfaceTexture: SurfaceTexture? ->
                        previewFrameTexture = surfaceTexture
                        // Make the display view visible to start showing the preview. This triggers the
                        // SurfaceHolder.Callback added to (the holder of) arSceneView.
                        previewDisplayView!!.visibility = View.VISIBLE
                    }
                    cameraHelper.startCamera(activity, CameraHelper.CameraFacing.FRONT, null)*/

                    arSceneView!!.session!!.pause()
                    setRepeatingCaptureRequest()

                    methodChannel2.invokeMethod("onGetIrisLandmarks", "PROCESSOR: ${processor}, SURFACE: ${arSceneView!!.holder.surface.isValid}")
                    val map = call.arguments as HashMap<*, *>
                    val displayWidth = map["width"] as? Int
                    val displayHeight = map["height"] as? Int

                    arSceneView!!.holder!!.addCallback(object : SurfaceHolder.Callback {
                        override fun surfaceCreated(holder: SurfaceHolder?) {
                            processor!!.videoSurfaceOutput.setSurface(holder!!.surface);
                        }

                        override fun surfaceChanged(holder: SurfaceHolder?, format: Int, width: Int, height: Int) {
                            var viewSize = Size(width, height)
                            converter!!.setSurfaceTextureAndAttachToGLContext(previewFrameTexture!!, displayWidth!!, displayHeight!!)
                        }

                        override fun surfaceDestroyed(holder: SurfaceHolder?) {
                            processor!!.videoSurfaceOutput.setSurface(null)
                        }
                    })

                    previewFrameTexture = arSceneView!!.session!!.sharedCamera.surfaceTexture
                    if (previewFrameTexture != null) {
                        arSceneView!!.visibility = View.VISIBLE
                    }

                    val focalLength = arSceneView?.arFrame?.camera?.imageIntrinsics?.focalLength?.get(0)
                    if (focalLength != null) {
                        var focalLenghtSidePacket = processor!!.packetCreator.createFloat32(focalLength)
                        val inputSidePackets = mapOf<String, Packet>(FOCAL_LENGTH_STREAM_NAME to focalLenghtSidePacket!!)
                        methodChannel2.invokeMethod("onGetIrisLandmarks", "inputSidePacket: ${inputSidePackets[FOCAL_LENGTH_STREAM_NAME]}")
                        processor!!.setInputSidePackets(inputSidePackets)

                        processor!!.addPacketCallback(
                                OUTPUT_LANDMARKS_STREAM_NAME
                        ) { packet: Packet ->
                            methodChannel2.invokeMethod("onGetIrisLandmarks", "INSIDE PACKET CALLBACK")
                            val landmarksRaw: ByteArray = PacketGetter.getProtoBytes(packet)
                            try {
                                val landmarks: LandmarkProto.NormalizedLandmarkList = LandmarkProto.NormalizedLandmarkList.parseFrom(landmarksRaw)
                                methodChannel2.invokeMethod("onGetIrisLandmarks", getLandmarksDebugString(landmarks))
                            } catch (e: Exception) {
                                return@addPacketCallback
                            }
                        }
                        result.success(true)
                    } else {
                        result.error("noFocalLength", "Focal length of the camera not found", null)
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

    // Called when starting non-AR mode or switching to non-AR mode.
    // Also called when app starts in AR mode, or resumes in AR mode.
    private fun setRepeatingCaptureRequest() {
        /*try {*/
            captureSession?.setRepeatingRequest(
                    previewCaptureRequestBuilder!!.build(), cameraCaptureCallback, backgroundHandler)
/*        } catch (e: CameraAccessException) {

        }*/
    }

    private fun resumeARCore() {
        // Ensure that session is valid before triggering ARCore resume. Handles the case where the user
        // manually uninstalls ARCore while the app is paused and then resumes.
        if (arSceneView?.session == null) {
            return
        }
        if (!arcoreActive) {
            try {
                // To avoid flicker when resuming ARCore mode inform the renderer to not suppress rendering
                // of the frames with zero timestamp.
                // backgroundRenderer.suppressTimestampZeroRendering(false)
                // Resume ARCore.
                arSceneView!!.session!!.resume()
                arcoreActive = true

                // Set capture session callback while in AR mode.
                arSceneView!!.session!!.sharedCamera!!.setCaptureCallback(cameraCaptureCallback, backgroundHandler)
            } catch (e: CameraNotAvailableException) {
                return
            }
        }
    }

    private fun pauseARCore() {
        if (arcoreActive) {
            // Pause ARCore.
            arSceneView!!.session!!.pause()
            arcoreActive = false
        }
    }

    private fun resumeCamera2() {
        setRepeatingCaptureRequest()
        arSceneView!!.session!!.sharedCamera.surfaceTexture.setOnFrameAvailableListener(this)
    }

    // Start background handler thread, used to run callbacks without blocking UI thread.
    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("sharedCameraBackground")
        backgroundThread!!.start()
        backgroundHandler = Handler(backgroundThread!!.getLooper())
    }

    // Repeating camera capture session capture callback.
    private val cameraCaptureCallback: CaptureCallback = object : CaptureCallback() {
        override fun onCaptureCompleted(
                session: CameraCaptureSession,
                request: CaptureRequest,
                result: TotalCaptureResult) {
        }

        override fun onCaptureBufferLost(
                session: CameraCaptureSession,
                request: CaptureRequest,
                target: Surface,
                frameNumber: Long) {
        }

        override fun onCaptureFailed(
                session: CameraCaptureSession,
                request: CaptureRequest,
                failure: CaptureFailure) {
        }

        override fun onCaptureSequenceAborted(
                session: CameraCaptureSession, sequenceId: Int) {
        }
    }

    // Repeating camera capture session state callback.
    var cameraSessionStateCallback: CameraCaptureSession.StateCallback = object : CameraCaptureSession.StateCallback() {
        // Called when the camera capture session is first configured after the app
        // is initialized, and again each time the activity is resumed.
        override fun onConfigured(@NonNull session: CameraCaptureSession) {
            captureSession = session
            if (arMode) {
                setRepeatingCaptureRequest()
                // Note, resumeARCore() must be called in onActive(), not here.
            } else {
                // Calls `setRepeatingCaptureRequest()`.
                resumeCamera2()
            }
        }

        override fun onSurfacePrepared(session: CameraCaptureSession, surface: Surface) {

        }

        override fun onReady(session: CameraCaptureSession) {
        }

        override fun onActive(session: CameraCaptureSession) {

            if (arMode && !arcoreActive) {
                resumeARCore()
            }
            /*synchronized(this@SharedCameraActivity) {
                captureSessionChangesPossible = true
                this@SharedCameraActivity.notify()
            }*/
        }

        override fun onCaptureQueueEmpty(session: CameraCaptureSession) {

        }

        override fun onClosed(session: CameraCaptureSession) {

        }

        override fun onConfigureFailed(p0: CameraCaptureSession) {

        }
    }

    // Camera device state callback.
    private val cameraDeviceCallback: CameraDevice.StateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(cameraDevice2: CameraDevice) {

            cameraDevice = cameraDevice2
            createCameraPreviewSession()
        }

        override fun onClosed(cameraDevice2: CameraDevice) {
            cameraDevice = null

        }

        override fun onDisconnected(cameraDevice2: CameraDevice) {

            cameraDevice2?.close()
            cameraDevice = null
        }

        override fun onError(cameraDevice2: CameraDevice, error: Int) {

            cameraDevice2?.close()
            cameraDevice = null

        }
    }

    private fun createCameraPreviewSession() {
        try {
            // arSceneView!!.session!!.setCameraTextureName(backgroundRenderer.getTextureId())
            arSceneView!!.session!!.sharedCamera.surfaceTexture.setOnFrameAvailableListener(this)

            // Create an ARCore compatible capture request using `TEMPLATE_RECORD`.
            previewCaptureRequestBuilder = cameraDevice!!.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)

            // Build surfaces list, starting with ARCore provided surfaces.
            val surfaceList = arSceneView!!.session!!.sharedCamera.arCoreSurfaces

            // Add a CPU image reader surface. On devices that don't support CPU image access, the image
            // may arrive significantly later, or not arrive at all.
            surfaceList.add(cpuImageReader!!.surface)

            // Surface list should now contain three surfaces:
            // 0. sharedCamera.getSurfaceTexture()
            // 1. …
            // 2. cpuImageReader.getSurface()

            // Add ARCore surfaces and CPU image surface targets.
            /*for (surface in surfaceList) {
                previewCaptureRequestBuilder.addTarget(surface)
            }*/

            // Wrap our callback in a shared camera callback.
            val wrappedCallback: CameraCaptureSession.StateCallback = arSceneView!!.session!!.sharedCamera.createARSessionStateCallback(cameraSessionStateCallback, backgroundHandler)

            // Create camera capture session for camera preview using ARCore wrapped callback.
            cameraDevice!!.createCaptureSession(surfaceList, wrappedCallback, backgroundHandler)
        } catch (e: CameraAccessException) {
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

    fun worldToScreen(screenWidth: Int, screenHeight: Int, worldToCameraMatrix: FloatArray): DoubleArray {
        val origin = FloatArray(4)
        origin[0] = 0f;
        origin[1] = 0f;
        origin[2] = 0f;
        origin[3] = 1f;

        val ndcCoord = FloatArray(4)
        multiplyMV(ndcCoord, 0,  worldToCameraMatrix, 0,  origin, 0);

        if (ndcCoord[3] != 0.0f) {
            ndcCoord[0] = (ndcCoord[0]/ndcCoord[3]).toFloat();
            ndcCoord[1] = (ndcCoord[1]/ndcCoord[3]).toFloat();
        }

        val pos_2d = DoubleArray(2)
        pos_2d[0] = (screenWidth  * ((ndcCoord[0] + 1.0)/2.0));
        pos_2d[1] = (screenHeight * (( 1.0 - ndcCoord[1])/2.0));

        return pos_2d;
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

    private fun takeScreenshot(call: MethodCall, result: MethodChannel.Result) {
        try {
            // create bitmap screen capture

            // Create a bitmap the size of the scene view.
            val bitmap: Bitmap = Bitmap.createBitmap(arSceneView!!.width, arSceneView!!.height,
                    Bitmap.Config.ARGB_8888)

            // Create a handler thread to offload the processing of the image.
            val handlerThread = HandlerThread("PixelCopier")
            handlerThread.start()

            // Make the request to copy.
            PixelCopy.request(arSceneView!!, bitmap, { copyResult ->
                if (copyResult === PixelCopy.SUCCESS) {
                    try {
                        saveBitmapToDisk(bitmap)
                    } catch (e: IOException) {
                        e.printStackTrace();
                    }
                }
                handlerThread.quitSafely()
            }, Handler(handlerThread.getLooper()))

        } catch (e: Throwable) {
            // Several error may come out with file handling or DOM
            e.printStackTrace()
        }
        result.success(null)
    }

    @Throws(IOException::class)
    fun saveBitmapToDisk(bitmap: Bitmap):String {
        val now = "rawScreenshot"
        // android/data/com.hswo.mvc_2021.hswo_mvc_2021_flutter_ar/files/
        // activity.applicationContext.getFilesDir().toString() //doesnt work!!
        // Environment.getExternalStorageDirectory()
        // val mPath: String =  Environment.getExternalStorageDirectory().toString() + "/DCIM/" + now + ".jpg"
        val mPath: String =  activity.applicationContext.getExternalFilesDir(null).toString() + "/" + now + ".png"
        val mediaFile = File(mPath)
        debugLog(mediaFile.toString())
        //Log.i("path","fileoutputstream opened")
        //Log.i("path",mPath)
        val fileOutputStream = FileOutputStream(mediaFile)
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fileOutputStream)
        fileOutputStream.flush()
        fileOutputStream.close()

        return mPath as String
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
        startBackgroundThread()
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

    override fun onImageAvailable(imageReader: ImageReader) {
        val image: Image = imageReader.acquireLatestImage()
        if (image == null) {
            return
        }

        image.close()
    }

    override fun onFrameAvailable(p0: SurfaceTexture?) {
    }

}
