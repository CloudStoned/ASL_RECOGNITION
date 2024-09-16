package com.example.asl_recog;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageProxy;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class DataCollector {
    private static final String TAG = "DataCollector";
    private final Context context;
    private String className;
    private int datasetSize;
    private int currentImageCount;

    public interface DataCollectionCallback {
        void onProgressUpdate(int progress, int total);
        void onCompleted();
    }

    private DataCollectionCallback callback;

    public DataCollector(Context context) {
        this.context = context;
    }

    public void setCallback(DataCollectionCallback callback) {
        this.callback = callback;
    }

    public void startDataCollection(String className, int datasetSize) {
        this.className = className;
        this.datasetSize = datasetSize;
        this.currentImageCount = 0;

        Log.d(TAG, "Data collection started for class: " + className + ", size: " + datasetSize);
    }

    public void processImage(@NonNull ImageProxy image) {
        if (currentImageCount >= datasetSize) {
            image.close();
            return;
        }

        Bitmap bitmap = imageProxyToBitmap(image);
        if (bitmap != null) {
            saveImageToGallery(bitmap);
            currentImageCount++;

            if (callback != null) {
                callback.onProgressUpdate(currentImageCount, datasetSize);
            }

            if (currentImageCount >= datasetSize && callback != null) {
                callback.onCompleted();
            }
        }

        image.close();
    }

    private Bitmap imageProxyToBitmap(ImageProxy image) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];

        // U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 100, out);

        byte[] imageBytes = out.toByteArray();
        Bitmap bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);

        // Rotate the bitmap to portrait orientation
        Matrix matrix = new Matrix();
        matrix.postRotate(90); // Rotate 90 degrees
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }

    private void saveImageToGallery(Bitmap bitmap) {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        String imageFileName = className + "_" + timeStamp + ".jpg";

        ContentResolver resolver = context.getContentResolver();
        ContentValues contentValues = new ContentValues();
        contentValues.put(MediaStore.MediaColumns.DISPLAY_NAME, imageFileName);
        contentValues.put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg");

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            contentValues.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/ASL_Dataset/" + className);
        } else {
            String imagesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).toString();
            contentValues.put(MediaStore.MediaColumns.DATA, imagesDir + "/ASL_Dataset/" + className + "/" + imageFileName);
        }

        Uri imageUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues);

        try {
            if (imageUri != null) {
                OutputStream outputStream = resolver.openOutputStream(imageUri);
                if (outputStream != null) {
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream);
                    outputStream.close();
                    Log.d(TAG, "Saved image: " + imageUri.toString());
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "Error saving image", e);
            if (imageUri != null) {
                resolver.delete(imageUri, null, null);
            }
        }
    }

    public int getCurrentImageCount() {
        return currentImageCount;
    }

    public int getDatasetSize() {
        return datasetSize;
    }

    public String getClassName() {
        return className;
    }
}