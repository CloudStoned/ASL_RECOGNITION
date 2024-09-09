package com.example.asl_recog;

import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.OptIn;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.mediapipe.formats.proto.LandmarkProto;
import com.google.mediapipe.solutions.hands.HandLandmark;
import com.google.mediapipe.solutions.hands.Hands;
import com.google.mediapipe.solutions.hands.HandsOptions;
import com.google.mediapipe.solutions.hands.HandsResult;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MediaPipeHandDetection";
    private static final int REQUEST_CODE_PERMISSION = 101;
    private static final String[] REQUIRED_PERMISSION = new String[]{"android.permission.CAMERA"};

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private PreviewView previewView;
    private TextView textView;
    private Hands hands;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.cameraView);
        textView = findViewById(R.id.result_text); // Make sure you have this in your layout

        if (checkPermissions()) {
            initializeHandsAndCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSION, REQUEST_CODE_PERMISSION);
        }
    }

    private boolean checkPermissions() {
        for (String permission : REQUIRED_PERMISSION) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                initializeHandsAndCamera();
            } else {
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    private void initializeHandsAndCamera() {
        // Initialize MediaPipe Hands
        HandsOptions handsOptions = HandsOptions.builder()
                .setStaticImageMode(false)
                .setMaxNumHands(1)
                .setMinDetectionConfidence(0.5f)
                .build();
        hands = new Hands(this, handsOptions);

        hands.setErrorListener((message, e) -> Log.e(TAG, "MediaPipe Hands error: " + message));

        startCamera();
    }

    private void startCamera() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Error starting camera: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindPreview(ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder().build();
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy image) {
                Bitmap bitmap = imageProxyToBitmap(image);
                if (bitmap != null) {
                    long timestamp = System.currentTimeMillis();
                    hands.send(bitmap, timestamp);
                }
                image.close();
            }
        });
        hands.setResultListener(this::processHandsResult);

        cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
    }

    private void processHandsResult(HandsResult result) {
        runOnUiThread(() -> {
            if (result.multiHandLandmarks().isEmpty()) {
                textView.setText("No hands detected");
            } else {
                textView.setText("Hand detected");
            }
        });
    }


    private String getHandPosition(HandsResult result)
    {
        // Get the first detected hand's landmarks
        List<LandmarkProto.NormalizedLandmark> landmarks = result.multiHandLandmarks().get(0).getLandmarkList();

        // Calculate the center of the palm
        float centerX = (landmarks.get(HandLandmark.WRIST).getX() +
                landmarks.get(HandLandmark.MIDDLE_FINGER_MCP).getX()) / 2;
        float centerY = (landmarks.get(HandLandmark.WRIST).getY() +
                landmarks.get(HandLandmark.MIDDLE_FINGER_MCP).getY()) / 2;

        // Determine the position based on the center of the palm
        String horizontalPosition = centerX < 0.33 ? "Left" : (centerX > 0.66 ? "Right" : "Center");
        String verticalPosition = centerY < 0.33 ? "Top" : (centerY > 0.66 ? "Bottom" : "Middle");

        return verticalPosition + " " + horizontalPosition;
    }

    // Convert Image object to Bitmap
    private Bitmap imageToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return Bitmap.createBitmap(image.getWidth(), image.getHeight(), Bitmap.Config.ARGB_8888);
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
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (hands != null) {
            hands.close();
        }
    }
}