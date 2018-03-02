package org.tensorflow.demo;

import android.app.Activity;
import android.content.Context;
import android.graphics.Point;
import android.graphics.PointF;
import android.media.ImageReader;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;
import android.view.Surface;
import android.widget.Toast;
import android.view.View;
import android.speech.tts.TextToSpeech;

import com.google.atap.tangoservice.Tango;
import com.google.atap.tangoservice.Tango.OnTangoUpdateListener;
import com.google.atap.tangoservice.Tango.TangoUpdateCallback;
import com.google.atap.tangoservice.TangoCameraIntrinsics;
import com.google.atap.tangoservice.TangoConfig;
import com.google.atap.tangoservice.TangoCoordinateFramePair;
import com.google.atap.tangoservice.TangoErrorException;
import com.google.atap.tangoservice.TangoEvent;
import com.google.atap.tangoservice.TangoInvalidException;
import com.google.atap.tangoservice.TangoOutOfDateException;
import com.google.atap.tangoservice.TangoPointCloudData;
import com.google.atap.tangoservice.TangoPoseData;
import com.google.atap.tangoservice.TangoXyzIjData;
import com.google.atap.tangoservice.experimental.TangoImageBuffer;
import com.google.tango.depthinterpolation.TangoDepthInterpolation;
import com.google.tango.support.TangoPointCloudManager;
import com.google.tango.support.TangoSupport;
import com.google.tango.transformhelpers.TangoTransformHelper;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import java.util.concurrent.TimeUnit;
import java.lang.Math;
import java.util.Locale;

import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.tracking.MultiBoxTracker;
import org.tensorflow.demo.R;
import org.tensorflow.demo.DetectorActivity;

/**
 * Created by manjekarbudhai on 7/27/17.
 */

public class MainActivity extends CameraActivity  {

    private Tango tango_;
    private TangoConfig tangoConfig_;
    private volatile boolean tangoConnected_ = false;
    private TangoPointCloudManager mPointCloudManager;
    HashMap<Integer, Integer> cameraTextures_ = null;
    private GLSurfaceView view_;
    private Renderer renderer_;
    private volatile TangoImageBuffer mCurrentImageBuffer;
    private int mDisplayRotation = 0;
    private Matrix rgbImageToDepthImage;
    TextToSpeech tts1;


    private class MeasuredPoint {
        public double mTimestamp;
        public float[] mDepthTPoint;

        public MeasuredPoint(double timestamp, float[] depthTPoint) {
            mTimestamp = timestamp;
            mDepthTPoint = depthTPoint;
        }
    }

    @Override
    public void onCreate(Bundle savedInstanceState){
        // GLSurfaceView for RGB color camera


        tts1=new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status != TextToSpeech.ERROR) {
                    tts1.setLanguage(Locale.UK);
                }
            }
        });



        Display display = this.getWindowManager().getDefaultDisplay();
        mDisplayRotation = display.getRotation();
        switch (mDisplayRotation){
            case (0):
                mDisplayRotation = Surface.ROTATION_0;
                break;
            case (90):
                mDisplayRotation = Surface.ROTATION_90;
                break;
            case(180):
                mDisplayRotation = Surface.ROTATION_180;
            case(270):
                mDisplayRotation = Surface.ROTATION_270;
                break;
            default:
                mDisplayRotation = Surface.ROTATION_90;
        }

        rgbImageToDepthImage = ImageUtils.getTransformationMatrix(
                640, 480,
                1920, 1080,
                0, true);

        super.onCreate(savedInstanceState);

        cameraTextures_ = new HashMap<>();
        mPointCloudManager = new TangoPointCloudManager();

        // Request depth in the Tango config because otherwise frames
        // are not delivered.
        tango_ = new Tango(this, new Runnable(){
            @Override
            public void run(){
                synchronized (this) {
                    try {
                        tangoConfig_ = tango_.getConfig(TangoConfig.CONFIG_TYPE_DEFAULT);
                        tangoConfig_.putBoolean(TangoConfig.KEY_BOOLEAN_COLORCAMERA, true);
                        tangoConfig_.putBoolean(TangoConfig.KEY_BOOLEAN_AUTORECOVERY, true);
                        tangoConfig_.putBoolean(TangoConfig.KEY_BOOLEAN_DEPTH, true);
                        tangoConfig_.putInt(TangoConfig.KEY_INT_DEPTH_MODE, TangoConfig.TANGO_DEPTH_MODE_POINT_CLOUD);
                        tango_.connect(tangoConfig_);
                        startTango();
                        TangoSupport.initialize(tango_);
                        //cameraTextures_ = new HashMap<>();

                    } catch (TangoOutOfDateException e) {
                        Log.i("new Tango", "error in onCreate");
                    }
                }
            }
        });

        try {
            TimeUnit.SECONDS.sleep(2);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        view_ = (GLSurfaceView)findViewById(R.id.surfaceviewclass);
        //view_.setZOrderMediaOverlay(true);
        view_.setEGLContextClientVersion(2);
        view_.setDebugFlags(GLSurfaceView.DEBUG_CHECK_GL_ERROR);
        view_.setRenderer(renderer_ = new Renderer(this));
        view_.setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);

        detect.setup();
        Log.i("onCreate", "detection setup completed");

        new Thread(new RunDetection()).start();

    }

    @Override
    public void onStart(){
        Log.i("onStart " , "Main onStart");
        super.onStart();
    }

    @Override
    public void onResume() {
        Log.i("onResume ", "Main onResume called");
        super.onResume();
        //startTango();
        if (tango_ == null) {
            startActivityForResult(
                    Tango.getRequestPermissionIntent(Tango.PERMISSIONTYPE_MOTION_TRACKING),
                    Tango.TANGO_INTENT_ACTIVITYCODE);


            tango_ = new Tango(this, new Runnable() {

                @Override
                public void run() {
                    Log.i("onResume ", "new tango");
                    synchronized (this) {
                        try {
                            tangoConfig_ = tango_.getConfig(TangoConfig.CONFIG_TYPE_DEFAULT);
                            tangoConfig_.putBoolean(TangoConfig.KEY_BOOLEAN_COLORCAMERA, true);
                            tangoConfig_.putBoolean(TangoConfig.KEY_BOOLEAN_AUTORECOVERY, true);
                            tangoConfig_.putBoolean(TangoConfig.KEY_BOOLEAN_DEPTH, true);
                            tangoConfig_.putInt(TangoConfig.KEY_INT_DEPTH_MODE, TangoConfig.TANGO_DEPTH_MODE_POINT_CLOUD);
                            tango_.connect(tangoConfig_);
                            startTango();
                            TangoSupport.initialize(tango_);
                            //cameraTextures_ = new HashMap<>();

                        } catch (TangoOutOfDateException e) {
                            Log.i("new Tango", "error in onCreate");
                        }
                    }
                }
            });
        }
    }

    @Override
     public void onPause() {
        super.onPause();
        /*if(tts1 !=null){
            tts1.stop();
            tts1.shutdown();
        }*/
        synchronized (this) {
            try {
                if (tango_ != null) {
                    tango_.disconnect();
                    tangoConnected_ = false;
                }
            }
            catch (TangoErrorException e) {
                Toast.makeText(
                        this,
                        "Tango error: " + e.getMessage(),
                        Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    public synchronized void onStop() {
        Log.i("onStop " , "Main onStop");
        super.onStop();
    }

    public synchronized void attachTexture(final int cameraId, final int textureName) {
        if (textureName > 0) {
            cameraTextures_.put(cameraId, textureName);
            Log.i("MainActivity","added to textures buffer");
            // Link the texture with Tango if the texture changes after
            // Tango is connected. This generally doesn't happen but
            // technically could because they happen in separate
            // threads. Otherwise the link will be made in startTango().
            if (cameraTextures_ != null && tango_ != null) {
                if (tangoConnected_ && cameraTextures_.get(cameraId) != textureName) {
                    tango_.connectTextureId(cameraId, textureName);
                    Log.i("attachTexture",String.format("cameraId: %d  textureName: %s",cameraId,textureName));
                }
            }
        }
        else {
            cameraTextures_.remove(cameraId);
        }
    }

    public synchronized void updateTexture(int cameraId) {
        if (tangoConnected_) {
            try {
                tango_.updateTexture(cameraId);
            }
            catch (TangoInvalidException e) {
                e.printStackTrace();
            }
        }
    }

    public Point getCameraFrameSize(int cameraId) {
        // TangoCameraIntrinsics intrinsics = mTango.getCameraIntrinsics(cameraId);
        // return new Point(intrinsics.width, intrinsics.height);
        return new Point(640, 480);
        //   return new Point(1280, 720);
    }



    protected  void processImageRGBbytes(int[] rgbBytes ) {}



    @Override
    public void onSetDebug(final boolean debug){}


    private void startTango() {
        try {

            tangoConnected_ = true;
            Log.i("startTango", "Tango Connected");



            // Attach cameras to textures.
            synchronized(this) {
                for (Map.Entry<Integer, Integer> entry : cameraTextures_.entrySet()) {
                    tango_.connectTextureId(entry.getKey(), entry.getValue());
                    Log.i("attachTexture",String.format("cameraId: %d  textureName: %s",entry.getKey(),entry.getValue()));
                }
            }

            // Attach Tango listener.
            ArrayList<TangoCoordinateFramePair> framePairs = new ArrayList<>();
            framePairs.add(new TangoCoordinateFramePair(
                    TangoPoseData.COORDINATE_FRAME_START_OF_SERVICE,
                    TangoPoseData.COORDINATE_FRAME_DEVICE));
            tango_.connectListener(framePairs, new Tango.TangoUpdateCallback(){
                @Override
                public void onPointCloudAvailable(TangoPointCloudData pointCloud) {
                    mPointCloudManager.updatePointCloud(pointCloud);
                }

                @Override
                public void onPoseAvailable(TangoPoseData tangoPoseData) {
                }

                @Override
                public void onXyzIjAvailable(TangoXyzIjData tangoXyzIjData) {
                }
                @Override
                public void onTangoEvent(TangoEvent tangoEvent) {
                    //Log.i("TangoEvent", String.format("%s: %s", tangoEvent.eventKey, tangoEvent.eventValue));
                }
                @Override
                public void onFrameAvailable(int i) {
                    //Log.i("onFrameAvailabe", "Main onFrameAvailabe called");
                    Log.i("onFrameAvailable",String.format("cameraid: %d",i));
                    if (i == TangoCameraIntrinsics.TANGO_CAMERA_COLOR) {
                        // mColorCameraPreview.onFrameAvailable();
                        view_.requestRender();
                        if(renderer_.argbInt != null){
                            detect.argbInt = renderer_.argbInt;
                            detect.processPerFrame();
                        }
                    }
                }
            });

            tango_.experimentalConnectOnFrameListener(TangoCameraIntrinsics.TANGO_CAMERA_COLOR,
                    new Tango.OnFrameAvailableListener() {
                        @Override
                        public void onFrameAvailable(TangoImageBuffer tangoImageBuffer, int i) {
                            mCurrentImageBuffer = copyImageBuffer(tangoImageBuffer);
                           // Log.i("onFrame",String.format("Tango Image Size: %dx%d",
                               //     mCurrentImageBuffer.width,mCurrentImageBuffer.height));
                        }

                        TangoImageBuffer copyImageBuffer(TangoImageBuffer imageBuffer) {
                            ByteBuffer clone = ByteBuffer.allocateDirect(imageBuffer.data.capacity());
                            imageBuffer.data.rewind();
                            clone.put(imageBuffer.data);
                            imageBuffer.data.rewind();
                            clone.flip();
                            return new TangoImageBuffer(imageBuffer.width, imageBuffer.height,
                                    imageBuffer.stride, imageBuffer.frameNumber,
                                    imageBuffer.timestamp, imageBuffer.format, clone,
                                    imageBuffer.exposureDurationNs);
                        }
                    });
        }
        catch (TangoOutOfDateException e) {
            Toast.makeText(
                    this,
                    "TangoCore update required",
                    Toast.LENGTH_SHORT).show();
        }
        catch (TangoErrorException e) {
            Toast.makeText(
                    this,
                    "Tango error: " + e.getMessage(),
                    Toast.LENGTH_SHORT).show();
        }
    }

    private  void stopTango() {
        try {
            if (tangoConnected_) {
                tango_.disconnect();
                tangoConnected_ = false;
            }
        }
        catch (TangoErrorException e) {
            Toast.makeText(
                    this,
                    "Tango error: " + e.getMessage(),
                    Toast.LENGTH_SHORT).show();
        }
    }

    public class RunDetection implements Runnable{
        @Override
        public void run(){
            final int  sleepShort = 5;
            int count = 0;
            int head_count = 0;
            float closest_depth;
            PointF closest_obstacle = new PointF(0.0f,0.0f);
            while(true) {
                try {
                    if(tangoConnected_ == false){
                        Thread.sleep(sleepShort);
                        continue;
                    }
                    if (null != renderer_.argbInt) {
                        ++count;
                        detect.argbInt = renderer_.argbInt;
                        detect.process();
                            if(count == 5) {
                                count = 0;
                                detect.processRects();
                                head_count = 0;
                                closest_depth = 10.0f;
                                closest_obstacle.set(0.0f,0.0f);
                                for(PointF rect: detect.rectDepthxy) {
                                    ++head_count;
                                    MeasuredPoint m = getBboxDepth(rect.x,rect.y);
                                    if (m.mDepthTPoint.length == 3) {
                                        if(m.mDepthTPoint[2] < closest_depth) {
                                            closest_depth = m.mDepthTPoint[2];
                                            closest_obstacle = new PointF(rect.x,rect.y);

                                        }
                                    }
                                }
                                if(closest_obstacle.x != 0.0f) {
                                    boolean orientation = getOrientationDir(closest_obstacle);
                                    double orientationval = getOrientationVal(closest_obstacle);
                                    if(orientationval > 30.0f && orientation) {
                                        tts1.speak(String.format("There are %d people. The closest is %d meters away, to your right.", head_count, Math.round(closest_depth)), TextToSpeech.QUEUE_FLUSH, null, "Detected");
                                    }
                                    else if(orientationval > 30.0f && !orientation){
                                        tts1.speak(String.format("There are %d people. The closest is %d meters away, to your left.", head_count, Math.round(closest_depth)), TextToSpeech.QUEUE_FLUSH, null, "Detected");
                                    }
                                    else{
                                        tts1.speak(String.format("There are %d people. The closest is %d meters away, in front of you.", head_count, Math.round(closest_depth)), TextToSpeech.QUEUE_FLUSH, null, "Detected");
                                    }
                                }
                            }
                        //detect.processRects();
                    } else {
                        Thread.sleep(sleepShort);
                    }
                }catch(Exception e){
                    System.out.println(e);
                }
            }
        }
    }


    public MeasuredPoint getBboxDepth(float u, float v) {
        TangoPointCloudData pointCloud = mPointCloudManager.getLatestPointCloud();
        if (pointCloud == null) {
            return null;
        }

        double rgbTimestamp;
        TangoImageBuffer imageBuffer = mCurrentImageBuffer;
        rgbTimestamp = imageBuffer.timestamp;

        TangoPoseData depthlTcolorPose = TangoSupport.getPoseAtTime(
                rgbTimestamp,
                TangoPoseData.COORDINATE_FRAME_CAMERA_DEPTH,
                TangoPoseData.COORDINATE_FRAME_CAMERA_COLOR,
                TangoSupport.ENGINE_TANGO,
                TangoSupport.ENGINE_TANGO,
                TangoSupport.ROTATION_IGNORED);
        if (depthlTcolorPose.statusCode != TangoPoseData.POSE_VALID) {
            Log.w("getdepthBbox", "Could not get color camera transform at time "
                    + rgbTimestamp);
            return null;
        }

        float[] depthPoint;


        depthPoint = TangoDepthInterpolation.getDepthAtPointBilateral(
                pointCloud,
                new double[] {0.0, 0.0, 0.0},
                new double[] {0.0, 0.0, 0.0, 1.0},
                imageBuffer,
                u, v,
                mDisplayRotation,
                depthlTcolorPose.translation,
                depthlTcolorPose.rotation);

        if (depthPoint == null) {
            Log.i("getBboxDepth()", "depth is null");
            return null;
        }
        //Log.i("getBboxDepth()", String.format("x:%f, y:%f, z:%f",depthPoint[0],depthPoint[1],depthPoint[2]));
        //tts1.speak("Depth detected",TextToSpeech.QUEUE_ADD,null,"Detected");
        return new MeasuredPoint(rgbTimestamp, depthPoint);
    }

    public boolean getOrientationDir(PointF bbox_in){
        boolean isClockwise = false;
        float adjacent = 320.0f - 640.0f*bbox_in.x;
        if(adjacent < 0.f){
            isClockwise = true;
        }
        else{
            isClockwise = false;
        }
        return isClockwise;
    }

    public double getOrientationVal(PointF bbox_in){
       double orientation = 0;
        double adjacent = (double)(480.0f - 480.0f*bbox_in.y);
        double opposite = (double)(Math.abs((320.0f - 640.0f*bbox_in.x)));
            orientation = Math.toDegrees(Math.atan(opposite/adjacent));
        return orientation;
    }
}



