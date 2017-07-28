package org.tensorflow.demo;

import android.content.Context;
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
import android.os.Bundle;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;
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

public class MainActivity extends CameraActivity implements ImageReader.OnImageAvailableListener {

    public void onPreviewSizeChosen(final Size size, final int rotation){
        detect.onPreviewSizeChosen(
                size,rotation);
    }

    public void onImageAvailable(final ImageReader reader){
        detect.onImageAvailable(reader);
    }

    protected  void processImageRGBbytes(int[] rgbBytes ) {}

    @Override
    protected int getLayoutId() {
       return detect.getLayoutId();
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return detect.getDesiredPreviewFrameSize();
    }

    @Override
    public void onSetDebug(final boolean debug){}
}
