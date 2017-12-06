/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.PointF;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.hardware.Camera;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;
import android.view.KeyEvent;

import java.nio.ByteBuffer;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;
import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.tracking.MultiBoxTracker;
import org.tensorflow.demo.R;
import org.tensorflow.demo.tracking.ObjectTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends Activity  {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged multibox model.
  private static final int MB_INPUT_SIZE = 224;
  private static final int MB_IMAGE_MEAN = 128;
  private static final float MB_IMAGE_STD = 128;
  private static final String MB_INPUT_NAME = "ResizeBilinear";
  private static final String MB_OUTPUT_LOCATIONS_NAME = "output_locations/Reshape";
  private static final String MB_OUTPUT_SCORES_NAME = "output_scores/Reshape";
  private static final String MB_MODEL_FILE = "file:///android_asset/multibox_model.pb";
  private static final String MB_LOCATION_FILE =
      "file:///android_asset/multibox_location_priors.txt";

  // Configuration values for tiny-yolo-voc. Note that the graph is not included with TensorFlow and
  // must be manually placed in the assets/ directory by the user.
  // Graphs and models downloaded from http://pjreddie.com/darknet/yolo/ may be converted e.g. via
  // DarkFlow (https://github.com/thtrieu/darkflow). Sample command:
  // ./flow --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights --savepb --verbalise
  private static final String YOLO_MODEL_FILE = "file:///android_asset/tiny-yolo-voc-1c.pb";
  private static final int YOLO_INPUT_SIZE = 416;
  private static final String YOLO_INPUT_NAME = "input";
  private static final String YOLO_OUTPUT_NAMES = "output";
  private static final int YOLO_BLOCK_SIZE = 32;

  // Default to the included multibox model.
  private static final boolean USE_YOLO = true;

  private static final int CROP_SIZE = USE_YOLO ? YOLO_INPUT_SIZE : MB_INPUT_SIZE;

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE = USE_YOLO ? 0.50f : 0.1f;

  private static final boolean MAINTAIN_ASPECT = USE_YOLO;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;

  private Integer sensorOrientation;

  private Classifier detector;

  private int previewWidth = 0;
  private int previewHeight = 0;
  private byte[][] yuvBytes;
  private int[] rgbBytes = null;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;

  private boolean computing = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private Bitmap cropCopyBitmap;

  private MultiBoxTracker tracker;

  private byte[] luminance;

  private BorderedText borderedText;

  private long lastProcessingTimeMs;
  private Context context;
  private boolean debug = false;
  private Handler handler;
  private HandlerThread handlerThread;

  protected Runnable postInferenceCallback;

  public volatile int[] argbInt = null;

  public List<RectF> rectDepth = new LinkedList<RectF>();
  public List<PointF> rectDepthxy = new LinkedList<PointF>();
  private boolean processingRects = false;
  private Matrix rgbImageToDepthImage = ImageUtils.getTransformationMatrix(
          640, 480,
          1920, 1080,
          0, true);

  public DetectorActivity(Context context){
    this.context = context;
  }

  /*public void addCallback(final OverlayView.DrawCallback callback) {
    final OverlayView overlay = (OverlayView) ((Activity)context).findViewById(R.id.debug_overlay);
    if (overlay != null) {
      overlay.addCallback(callback);
    }
  }*/

  protected void runInBackground(final Runnable r) {

    handlerThread = new HandlerThread("inference");
    handlerThread.start();

    handler = new Handler(handlerThread.getLooper());
    if (handler != null) {
      handler.post(r);
    }
  }

  public boolean isDebug() {
    return debug;
  }

  protected void fillBytes(final Plane[] planes, final byte[][] yuvBytes) {
    // Because of the variable row stride it's not possible to know in
    // advance the actual necessary dimensions of the yuv planes.
    for (int i = 0; i < planes.length; ++i) {
      final ByteBuffer buffer = planes[i].getBuffer();
      if (yuvBytes[i] == null) {
        LOGGER.d("Initializing buffer %d at size %d", i, buffer.capacity());
        yuvBytes[i] = new byte[buffer.capacity()];
      }
      buffer.get(yuvBytes[i]);
    }
  }

  /*@Override
  public void onPreviewFrame(final byte[] bytes, final Camera camera) {
    if (computing) {
      return;
    }
    computing = true;
    yuvBytes[0] = bytes;
    try {
      // Initialize the storage bitmaps once when the resolution is known.
      if (rgbBytes == null) {
        Camera.Size previewSize = camera.getParameters().getPreviewSize();
        previewHeight = previewSize.height;
        previewWidth = previewSize.width;
        rgbBytes = new int[previewWidth * previewHeight];
        onPreviewSizeChosen(new Size(previewSize.width, previewSize.height), 90);
      }
      ImageUtils.convertYUV420SPToARGB8888(bytes, rgbBytes, previewWidth, previewHeight, false);
    } catch (final Exception e) {
      LOGGER.e(e, "Exception!");
      return;
    }
    postInferenceCallback = new Runnable() {
      @Override
      public void run() {
        camera.addCallbackBuffer(bytes);
      }
    };
    processImageRGBbytes(rgbBytes);
  }*/

  @Override
  public boolean onKeyDown(final int keyCode, final KeyEvent event) {
    if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN || keyCode == KeyEvent.KEYCODE_VOLUME_UP) {
      debug = !debug;
      requestRender();
      onSetDebug(debug);
      return true;
    }
    return super.onKeyDown(keyCode, event);
  }


  public void setup() {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(context);


    if (USE_YOLO) {
      detector =
          TensorFlowYoloDetector.create(
              context.getAssets(),
              YOLO_MODEL_FILE,
              YOLO_INPUT_SIZE,
              YOLO_INPUT_NAME,
              YOLO_OUTPUT_NAMES,
              YOLO_BLOCK_SIZE);
    } else {
      detector =
          TensorFlowMultiBoxDetector.create(
              getAssets(),
              MB_MODEL_FILE,
              MB_LOCATION_FILE,
              MB_IMAGE_MEAN,
              MB_IMAGE_STD,
              MB_INPUT_NAME,
              MB_OUTPUT_LOCATIONS_NAME,
              MB_OUTPUT_SCORES_NAME);
    }

    previewWidth = DESIRED_PREVIEW_SIZE.getWidth();
    previewHeight = DESIRED_PREVIEW_SIZE.getHeight();
    argbInt = new int[previewWidth * previewHeight];

   // final Display display = ((Activity)context).getWindowManager().getDefaultDisplay();
   // final int screenOrientation = display.getRotation();

   // LOGGER.i("Sensor orientation: %d, Screen orientation: %d", rotation, screenOrientation);

    //sensorOrientation = rotation + screenOrientation;
    sensorOrientation = 0;

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbBytes = new int[previewWidth * previewHeight];
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(CROP_SIZE, CROP_SIZE, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            CROP_SIZE, CROP_SIZE,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);
    yuvBytes = new byte[3][];

    trackingOverlay = (OverlayView) ((Activity)context).findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            /*tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }*/
            tracker.drawYolo(canvas);
            if(!tracker.rectDepth.isEmpty() && !processingRects) {
              rectDepth = tracker.rectDepth;
              //LOGGER.i("rect(0): %s", tracker.rectDepth.get(0));
            }
          }
        });

  }

  public void processRects(){
    processingRects = true;
    rectDepthxy.clear();
    //LOGGER.i("Orig coord: %s",rectDepth.get(0));
    for(RectF rect : rectDepth){
      //LOGGER.i("Orig coord: %s",rect);
      rgbImageToDepthImage.mapRect(rect);
      rectDepthxy.add(new PointF(rect.centerX()/1920.0f,rect.centerY()/1080.0f));
    }
    processingRects = false;
    //LOGGER.i("Normalized coord: %s",rectDepthxy.get(0));
  }

  public void requestRender() {
    final OverlayView overlay = (OverlayView) ((Activity)context).findViewById(R.id.debug_overlay);
    if (overlay != null) {
      overlay.postInvalidate();
    }
  }

  OverlayView trackingOverlay;

  public void processPerFrame(){

    ++timestamp;
    //long timeStart = SystemClock.uptimeMillis();
    int previewSize = previewWidth * previewHeight;
    byte [] yuv420sp = new byte[(previewSize * 3) >>> 1];
    //byte [] y = new byte[previewSize];
    ImageUtils.convertARGB8888ToYUV420SP(argbInt, yuv420sp, previewWidth, previewHeight);
   /* for(int i = 0; i < previewSize; ++i){
      y[i] = yuv420sp[i];
    } */

    tracker.onFrame(
            previewWidth,
            previewHeight,
            640,
            sensorOrientation,
            yuv420sp,
            timestamp);
    trackingOverlay.postInvalidate();

    //long processTime = SystemClock.uptimeMillis() - timeStart;
    //LOGGER.i("process time: %d",processTime);

  }

  public void process() {
    /*if (0 == timestamp % 100) {
      LOGGER.w("onImageAvailable(): [%d] Width x Height = [%d x %d]",
              timestamp, DESIRED_PREVIEW_SIZE.getWidth(), DESIRED_PREVIEW_SIZE.getHeight());
    }*/

    //LOGGER.i("Process() started %d", timestamp);
    //Image image = null;

    //++timestamp;
    final long currTimestamp = timestamp;
/*
    try {
      image = reader.acquireLatestImage();

      if (image == null) {
        return;
      }

      Trace.beginSection("imageAvailable");

      final Plane[] planes = image.getPlanes();
      fillBytes(planes, yuvBytes);

      tracker.onFrame(
          previewWidth,
          previewHeight,
          planes[0].getRowStride(),
          sensorOrientation,
          yuvBytes[0],
          timestamp);
      trackingOverlay.postInvalidate();

      // No mutex needed as this method is not reentrant.
      if (computing) {
        image.close();
        return;
      }*/
    /*int previewSize = previewWidth * previewHeight;
    byte [] yuv420sp = new byte[(previewSize * 3) >>> 1];
    ImageUtils.convertARGB8888ToYUV420SP(argbInt, yuv420sp, previewWidth, previewHeight);*/

   /* for (int m = 0; m < yuv420sp.length; m++) {
      if (yuv420sp[m] < 0) {
        int tmp = 256 + yuv420sp[m];
        tmp /= 2;
        yuv420sp[m] = (byte) tmp;
      } else {
        yuv420sp[m] /= 2;
      }
    }*/

   /* tracker.onFrame(
            previewWidth,
            previewHeight,
            640,
            sensorOrientation,
            yuv420sp,
            timestamp);
    trackingOverlay.postInvalidate();*/

    if (computing) {
      return;
    }
      computing = true;

      /*final int yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();
      ImageUtils.convertYUV420ToARGB8888(
              yuvBytes[0],
              yuvBytes[1],
              yuvBytes[2],
              rgbBytes,
              previewWidth,
              previewHeight,
              yRowStride,
              uvRowStride,
              uvPixelStride,
              false);


      image.close();
    } catch (final Exception e) {
      if (image != null) {
        image.close();
      }
      LOGGER.e(e, "Exception!");
      Trace.endSection();
      return;
    }*/

    int previewSize = previewWidth * previewHeight;
    byte [] yuv420sp = new byte[(previewSize * 3) >>> 1];
    ImageUtils.convertARGB8888ToYUV420SP(argbInt, yuv420sp, previewWidth, previewHeight);
/*
    for (int m = 0; m < yuv420sp.length; m++) {
      if (yuv420sp[m] < 0) {
        int tmp = 256 + yuv420sp[m];
        tmp /= 2;
        yuv420sp[m] = (byte) tmp;
      } else {
        yuv420sp[m] /= 2;
      }
    }
*/
    /*tracker.onFrame(
            previewWidth,
            previewHeight,
            640,
            sensorOrientation,
            yuv420sp,
            timestamp);
    trackingOverlay.postInvalidate();*/

    rgbFrameBitmap.setPixels(argbInt, 0, previewWidth, 0, 0, previewWidth, previewHeight);
    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    if (luminance == null) {
      //luminance = new byte[yuvBytes[0].length];
      luminance = new byte[previewSize];
    }
    //System.arraycopy(yuvBytes[0], 0, luminance, 0, luminance.length);
    System.arraycopy(yuv420sp, 0, luminance, 0, luminance.length);

    /*runInBackground(
        new Runnable() {
          @Override
          public void run() {*/
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            /*cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);*/

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE) {
                //canvas.drawRect(location, paint);
                LOGGER.i("(416x416: ",location);
                cropToFrameTransform.mapRect(location);
                result.setLocation(location);
                mappedRecognitions.add(result);
              }
            }

            tracker.trackResults(mappedRecognitions, luminance, currTimestamp);
            trackingOverlay.postInvalidate();

            requestRender();
            computing = false;
          }
      /*  });

    Trace.endSection();
  }*/

  //protected  void processImageRGBbytes(int[] rgbBytes ) {}


  // protected int getLayoutId() {return R.layout.camera_connection_fragment_tracking;}


  //protected Size getDesiredPreviewFrameSize() {return DESIRED_PREVIEW_SIZE;}


  public void onSetDebug(final boolean debug) {
    detector.enableStatLogging(debug);
  }
}
