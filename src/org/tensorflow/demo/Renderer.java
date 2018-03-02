// Copyright 2015 Shoestring Research, LLC.  All rights reserved.
package org.tensorflow.demo;

import android.graphics.Point;
import android.opengl.GLES11Ext;
import android.opengl.GLSurfaceView;
import android.util.Log;

import com.google.atap.tangoservice.Tango;
import com.google.atap.tangoservice.TangoCameraIntrinsics;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import static android.opengl.GLES20.GL_ARRAY_BUFFER;
import static android.opengl.GLES20.GL_BYTE;
import static android.opengl.GLES20.GL_COLOR_ATTACHMENT0;
import static android.opengl.GLES20.GL_FRAGMENT_SHADER;
import static android.opengl.GLES20.GL_FRAMEBUFFER;
import static android.opengl.GLES20.GL_LINEAR;
import static android.opengl.GLES20.GL_NEAREST;
import static android.opengl.GLES20.GL_RENDERBUFFER;
import static android.opengl.GLES20.GL_RGBA;
import static android.opengl.GLES20.GL_STATIC_DRAW;
import static android.opengl.GLES20.GL_TEXTURE0;
import static android.opengl.GLES20.GL_TEXTURE_MAG_FILTER;
import static android.opengl.GLES20.GL_TEXTURE_MIN_FILTER;
import static android.opengl.GLES20.GL_TRIANGLE_STRIP;
import static android.opengl.GLES20.GL_UNSIGNED_BYTE;
import static android.opengl.GLES20.GL_VERTEX_SHADER;
import static android.opengl.GLES20.GL_VIEWPORT;
import static android.opengl.GLES20.glActiveTexture;
import static android.opengl.GLES20.glAttachShader;
import static android.opengl.GLES20.glBindBuffer;
import static android.opengl.GLES20.glBindFramebuffer;
import static android.opengl.GLES20.glBindRenderbuffer;
import static android.opengl.GLES20.glBindTexture;
import static android.opengl.GLES20.glBufferData;
import static android.opengl.GLES20.glClearColor;
import static android.opengl.GLES20.glCompileShader;
import static android.opengl.GLES20.glCreateProgram;
import static android.opengl.GLES20.glCreateShader;
import static android.opengl.GLES20.glDrawArrays;
import static android.opengl.GLES20.glEnableVertexAttribArray;
import static android.opengl.GLES20.glFramebufferRenderbuffer;
import static android.opengl.GLES20.glGenBuffers;
import static android.opengl.GLES20.glGenFramebuffers;
import static android.opengl.GLES20.glGenRenderbuffers;
import static android.opengl.GLES20.glGenTextures;
import static android.opengl.GLES20.glGetAttribLocation;
import static android.opengl.GLES20.glGetIntegerv;
import static android.opengl.GLES20.glGetProgramInfoLog;
import static android.opengl.GLES20.glGetShaderInfoLog;
import static android.opengl.GLES20.glGetUniformLocation;
import static android.opengl.GLES20.glLinkProgram;
import static android.opengl.GLES20.glReadPixels;
import static android.opengl.GLES20.glRenderbufferStorage;
import static android.opengl.GLES20.glShaderSource;
import static android.opengl.GLES20.glTexParameteri;
import static android.opengl.GLES20.glUniform1i;
import static android.opengl.GLES20.glUseProgram;
import static android.opengl.GLES20.glVertexAttribPointer;
import static android.opengl.GLES20.glViewport;
import static android.opengl.GLES30.GL_MAP_INVALIDATE_BUFFER_BIT;
import static android.opengl.GLES30.GL_MAP_WRITE_BIT;
import static android.opengl.GLES30.GL_RGBA8;
import static android.opengl.GLES30.glMapBufferRange;
import static android.opengl.GLES30.glUnmapBuffer;

// Need OpenGL ES 3.0 for RGBA8 renderbuffer.

class Renderer implements GLSurfaceView.Renderer {
   // Vertex program flips Y when drawing on screen, doesn't flip
   // when drawing offscreen for saving.
   static final String videoVertexSource =
      "uniform mediump int cap;\n" +
      "attribute vec4 a_v;\n" +
      "varying vec2 t;\n" +
      "void main() {\n" +
      "	gl_Position = a_v;\n" +
      "	t = 0.5*vec2(a_v.x, cap != 0 ? a_v.y : -a_v.y) + vec2(0.5,0.5);\n" +
      "}\n";

   // Fragment buffer reorders color components when drawing offscreen
   // for saving.
   static final String videoFragmentSource =
      "#extension GL_OES_EGL_image_external : require\n" +
      "precision mediump float;\n" +
      "uniform mediump int cap;\n" +
      "varying vec2 t;\n" +
      "uniform samplerExternalOES colorTex;\n" +
      "void main() {\n" +
      "  vec4 c = texture2D(colorTex, t);\n" +
      "	gl_FragColor = cap != 0 ? c.bgra : c;\n" +
      "}\n";

   MainActivity activity_;

   int videoProgram_;
   int videoVertexAttribute_;
   int videoVertexBuffer_;
   int videoTextureName_;

   int offscreenBuffer_;
   Point offscreenSize_;

   volatile boolean saveNextFrame_;
   public volatile int[] argbInt = null;

   Renderer(MainActivity activity) {
      activity_ = activity;
   }

   @Override
   public void onSurfaceCreated(GL10 gl, EGLConfig config) {
      glClearColor(0.3f, 0.3f, 0.3f, 1.0f);

      IntBuffer bufferNames = IntBuffer.allocate(1);
      glGenBuffers(1, bufferNames);
      videoVertexBuffer_ = bufferNames.get(0);

      // Create a bi-unit square geometry.
      glBindBuffer(GL_ARRAY_BUFFER, videoVertexBuffer_);
      glBufferData(GL_ARRAY_BUFFER, 8, null, GL_STATIC_DRAW);
      ((ByteBuffer)glMapBufferRange(
         GL_ARRAY_BUFFER,
         0, 8,
         GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT))
         .order(ByteOrder.nativeOrder())
         .put(new byte[] { -1, 1,  -1, -1,  1, 1,  1, -1 });
      glUnmapBuffer(GL_ARRAY_BUFFER);

      // Create the video texture.
      IntBuffer textureNames = IntBuffer.allocate(1);
      glGenTextures(1, textureNames);
      videoTextureName_ = textureNames.get(0);

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, videoTextureName_);

      // Connect the texture to Tango.
      activity_.attachTexture(TangoCameraIntrinsics.TANGO_CAMERA_COLOR, videoTextureName_);
      Log.i("Renderer",String.format("cameraid: %d texturename: %d", TangoCameraIntrinsics.TANGO_CAMERA_COLOR,videoTextureName_));

      // Prepare the shader program.
      videoProgram_ = createShaderProgram(videoVertexSource, videoFragmentSource);
      glUseProgram(videoProgram_);
      videoVertexAttribute_ = glGetAttribLocation(videoProgram_, "a_v");
      glUniform1i(
              glGetUniformLocation(videoProgram_, "colorTex"),
              0);  // GL_TEXTURE0
      glUniform1i(
              glGetUniformLocation(videoProgram_, "cap"),
              0);

      // Get the camera frame dimensions.
      offscreenSize_ = activity_.getCameraFrameSize(TangoCameraIntrinsics.TANGO_CAMERA_COLOR);

      // Create an offscreen render target to capture a frame.
      IntBuffer renderbufferName = IntBuffer.allocate(1);
      glGenRenderbuffers(1, renderbufferName);
      glBindRenderbuffer(GL_RENDERBUFFER, renderbufferName.get(0));
      glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, offscreenSize_.x, offscreenSize_.y);

      IntBuffer framebufferName = IntBuffer.allocate(1);
      glGenFramebuffers(1, framebufferName);
      glBindFramebuffer(GL_FRAMEBUFFER, framebufferName.get(0));
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, renderbufferName.get(0));

      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      offscreenBuffer_ = framebufferName.get(0);
   }

   @Override
   public void onSurfaceChanged(GL10 gl, int width, int height) {
      glViewport(0, 0, width, height);
   }

   /* public static int[] decodeYUV420SPtoRGB(byte[] yuv420sp, int width, int height) {
      if (yuv420sp == null) throw new NullPointerException();

      final int frameSize = width * height;
      int[] rgb = new int[frameSize];

      for (int j = 0, yp = 0; j < height; j++) {
         int uvp = frameSize + (j >> 1) * width, u = 0, v = 0;
         for (int i = 0; i < width; i++, yp++) {
            int y = (0xff & (yuv420sp[yp])) - 16;
            if (y < 0) y = 0;
            if ((i & 1) == 0) {
               v = (0xff & yuv420sp[uvp++]) - 128;
               u = (0xff & yuv420sp[uvp++]) - 128;
            }
            int y1192 = 1192 * y;
            int r = (y1192 + 1634 * v);
            int g = (y1192 - 833 * v - 400 * u);
            int b = (y1192 + 2066 * u);

            if (r < 0) r = 0;
            else if (r > 262143) r = 262143;
            if (g < 0) g = 0;
            else if (g > 262143) g = 262143;
            if (b < 0) b = 0;
            else if (b > 262143) b = 262143;

            rgb[yp] = 0xff000000 | ((r << 6) & 0xff0000) | ((g >> 2) & 0xff00) | ((b >> 10) & 0xff);
         }
      }
      return rgb;
   }  */

   private static byte[] decodeYUV420SPtoY(byte[] yuv420sp, int width, int height) {
      if (yuv420sp == null) throw new NullPointerException();

      final int frameSize = width * height;
      byte[] yAll = new byte[frameSize];
      byte[] uAll = new byte[frameSize];
      byte[] vAll = new byte[frameSize];

      int yp = 0;
      for (int j = 0; j < height; j++) {
         int uvp = frameSize + (j >> 1) * width, u = 0, v = 0;
         for (int i = 0; i < width; i++) {
            int y = (0xff & (yuv420sp[yp])) - 16;
            if (y < 0) y = 0;
            if ((i & 1) == 0) {
               v = (0xff & yuv420sp[uvp++]) - 128;
               u = (0xff & yuv420sp[uvp++]) - 128;
            }
            yAll[yp] = (byte)y;
            uAll[yp] = (byte)u;
            vAll[yp] = (byte)v;
            yp++;
         }
      }
      return yAll;
   }

   @Override
   public void onDrawFrame(GL10 gl) {
      activity_.updateTexture(TangoCameraIntrinsics.TANGO_CAMERA_COLOR);

      /* if(argbInt != null) {
           glBindBuffer(GL_ARRAY_BUFFER, videoVertexBuffer_);
           glVertexAttribPointer(videoVertexAttribute_, 2, GL_BYTE, false, 0, 0);
           glEnableVertexAttribArray(videoVertexAttribute_);
           glActiveTexture(GL_TEXTURE0);
           glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, videoTextureName_);
           glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
           glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
           glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

           return;
       }*/

      glBindBuffer(GL_ARRAY_BUFFER, videoVertexBuffer_);
      glVertexAttribPointer(videoVertexAttribute_, 2, GL_BYTE, false, 0, 0);
      glEnableVertexAttribArray(videoVertexAttribute_);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, videoTextureName_);
      glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

      // Switch to the offscreen buffer.
      glBindFramebuffer(GL_FRAMEBUFFER, offscreenBuffer_);

      // Save current viewport and change to offscreen size.
      IntBuffer viewport = IntBuffer.allocate(4);
      glGetIntegerv(GL_VIEWPORT, viewport);
      //glViewport(0, 0, offscreenSize_.x, offscreenSize_.y);
      glViewport(0, 0, offscreenSize_.x, offscreenSize_.y);

      // Render in capture mode. Setting this flags tells the shader
      // program to draw the texture right-side up and change the color
      // order to ARGB for compatibility with Bitmap.
      glUniform1i(
              glGetUniformLocation(videoProgram_, "cap"),
              1);

      // Render.
      glBindBuffer(GL_ARRAY_BUFFER, videoVertexBuffer_);
      glVertexAttribPointer(videoVertexAttribute_, 2, GL_BYTE, false, 0, 0);
      glEnableVertexAttribArray(videoVertexAttribute_);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, videoTextureName_);
      glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

      // Read offscreen buffer.
      IntBuffer intBuffer = ByteBuffer.allocateDirect(offscreenSize_.x * offscreenSize_.y * 4)
              .order(ByteOrder.nativeOrder())
              .asIntBuffer();
      glReadPixels(0, 0, offscreenSize_.x, offscreenSize_.y, GL_RGBA, GL_UNSIGNED_BYTE, intBuffer.rewind());

      // Restore onscreen state.
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glViewport(viewport.get(0), viewport.get(1), viewport.get(2), viewport.get(3));
      glUniform1i(
              glGetUniformLocation(videoProgram_, "cap"),
              0);

      // Convert to an array for Bitmap.createBitmap().
      int[] pixels = new int[intBuffer.capacity()];
      intBuffer.rewind();
      intBuffer.get(pixels);
/*
      glBindBuffer(GL_ARRAY_BUFFER, videoVertexBuffer_);
      glVertexAttribPointer(videoVertexAttribute_, 2, GL_BYTE, false, 0, 0);
      glEnableVertexAttribArray(videoVertexAttribute_);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, videoTextureName_);
      glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
*/
      // Image frame from Camera is like 30HZ
      // However, image detection & Tracking is processed like 2HZ

      // If the previous image buffer has been processed, argbInt is NULL

      /*int width = offscreenSize_.x;
      int height = offscreenSize_.y;
      byte [] yuv420sp = new byte[width * height * 3];
      ImageUtils.convertARGB8888ToYUV420SP(pixels, yuv420sp, width, height);
      byte [] yAll = decodeYUV420SPtoY(yuv420sp, width, height);

      mNewYData = yAll; */
      argbInt = pixels; // will be accessed from the processing task in NavigationActitivty

      /*
      if (!saveNextFrame_) {
         glBindBuffer(GL_ARRAY_BUFFER, videoVertexBuffer_);
         glVertexAttribPointer(videoVertexAttribute_, 2, GL_BYTE, false, 0, 0);
         glEnableVertexAttribArray(videoVertexAttribute_);
         glActiveTexture(GL_TEXTURE0);
         glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, videoTextureName_);
         glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
         glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
         glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      }
      else {
         // Switch to the offscreen buffer.
         glBindFramebuffer(GL_FRAMEBUFFER, offscreenBuffer_);

         // Save current viewport and change to offscreen size.
         IntBuffer viewport = IntBuffer.allocate(4);
         glGetIntegerv(GL_VIEWPORT, viewport);
         glViewport(0, 0, offscreenSize_.x, offscreenSize_.y);

         // Render in capture mode. Setting this flags tells the shader
         // program to draw the texture right-side up and change the color
         // order to ARGB for compatibility with Bitmap.
         glUniform1i(
            glGetUniformLocation(videoProgram_, "cap"),
            1);

         // Render.
         glBindBuffer(GL_ARRAY_BUFFER, videoVertexBuffer_);
         glVertexAttribPointer(videoVertexAttribute_, 2, GL_BYTE, false, 0, 0);
         glEnableVertexAttribArray(videoVertexAttribute_);
         glActiveTexture(GL_TEXTURE0);
         glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, videoTextureName_);
         glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
         glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
         glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

         // Read offscreen buffer.
         IntBuffer intBuffer = ByteBuffer.allocateDirect(offscreenSize_.x * offscreenSize_.y * 4)
                 .order(ByteOrder.nativeOrder())
                 .asIntBuffer();
         glReadPixels(0, 0, offscreenSize_.x, offscreenSize_.y, GL_RGBA, GL_UNSIGNED_BYTE, intBuffer.rewind());

         // Restore onscreen state.
         glBindFramebuffer(GL_FRAMEBUFFER, 0);
         glViewport(viewport.get(0), viewport.get(1), viewport.get(2), viewport.get(3));
         glUniform1i(
                 glGetUniformLocation(videoProgram_, "cap"),
                 0);

         // Convert to an array for Bitmap.createBitmap().
         int[] pixels = new int[intBuffer.capacity()];
         intBuffer.rewind();
         intBuffer.get(pixels);
         saveNextFrame_ = false;

         try {
            // Create/access a pictures subdirectory.
            File directory = new File(
               Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
               "Tango Captures");
            if (!directory.mkdirs() && !directory.isDirectory()) {
               Toast.makeText(
                       activity_,
                       "Could not access save directory",
                       Toast.LENGTH_SHORT).show();
               return;
            }

            // Get the current capture index to construct a unique filename.
            SharedPreferences prefs = activity_.getPreferences(Context.MODE_PRIVATE);
            int index = prefs.getInt("index", 0);
            SharedPreferences.Editor prefsEditor = prefs.edit();
            prefsEditor.putInt("index", index + 1);
            prefsEditor.commit();

            // Create the capture file.
            File file = new File(directory, String.format("tango%05d.png", index));
            FileOutputStream fileOutputStream = new FileOutputStream(file);

            // Bitmap conveniently provides file output.
            Bitmap bitmap = Bitmap.createBitmap(pixels, offscreenSize_.x, offscreenSize_.y, Bitmap.Config.ARGB_8888);
            bitmap.compress(Bitmap.CompressFormat.PNG, 90, fileOutputStream);
            fileOutputStream.close();

            // Make the new file visible to other apps.
            activity_.sendBroadcast(new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE, Uri.fromFile(file)));
         }
         catch (Exception e) {
            e.printStackTrace();
         }
      } */
   }

   void saveFrame() {
      saveNextFrame_ = true;
   }

   private int createShaderProgram(String vertexSource, String fragmentSource) {
      int vsName = glCreateShader(GL_VERTEX_SHADER);
      glShaderSource(vsName, vertexSource);
      glCompileShader(vsName);
      System.out.println(glGetShaderInfoLog(vsName));

      int fsName = glCreateShader(GL_FRAGMENT_SHADER);
      glShaderSource(fsName, fragmentSource);
      glCompileShader(fsName);
      System.out.println(glGetShaderInfoLog(fsName));

      int programName = glCreateProgram();
      glAttachShader(programName, vsName);
      glAttachShader(programName, fsName);
      glLinkProgram(programName);
      System.out.println(glGetProgramInfoLog(programName));

      return programName;
   }

}
