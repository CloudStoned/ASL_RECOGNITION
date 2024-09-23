package com.example.asl_recog;


import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;


import com.example.asl_recog.databinding.ActivityMainBinding;
import com.google.common.util.concurrent.ListenableFuture;

import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity implements Detector.DetectorListener {
    private static final String TAG = "MainActivity";
    private static final int REQUEST_CODE_PERMISSIONS = 10;
    private static final String[] REQUIRED_PERMISSIONS = {Manifest.permission.CAMERA};

    private ActivityMainBinding binding;
    private Detector detector;
    private ExecutorService cameraExecutor;
    private StringBuilder combinedLetters = new StringBuilder();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        cameraExecutor = Executors.newSingleThreadExecutor();

        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }

        initializeDetector();
        setupButtons();
    }

    private void initializeDetector() {
        cameraExecutor.execute(() -> {
            detector = new Detector(getApplicationContext(), "best_float32.tflite", "classes.txt", this);
        });
    }

    private void setupButtons() {
        binding.combineLetters.setOnClickListener(v -> {
            combinedLetters.append(binding.resultText.getText());
            binding.combinedLetters.setText(combinedLetters.toString());
        });

        binding.clearButton.setOnClickListener(v -> {
            combinedLetters.setLength(0);
            binding.combinedLetters.setText("");
            binding.resultText.setText("Result Here");
        });

        binding.collectDatasetButton.setOnClickListener(v -> {
            // Implement dataset collection logic here
            Log.d(TAG, "Dataset collection button clicked");
        });
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindCameraUseCases(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Error starting camera: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCameraUseCases(ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(cameraExecutor, this::analyze);

        preview.setSurfaceProvider(binding.cameraView.getSurfaceProvider());

        Camera camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
    }

    private void analyze(ImageProxy image) {
        if (detector == null) return;

        int rotationDegrees = image.getImageInfo().getRotationDegrees();
        Bitmap bitmap = imageToBitmap(image);
        bitmap = rotateBitmap(bitmap, rotationDegrees);

        detector.detect(bitmap);

        image.close();
    }

    private Bitmap imageToBitmap(ImageProxy image) {
        // Implementation of imageToBitmap method (as in the previous example)
        // ...
    }

    private Bitmap rotateBitmap(Bitmap bitmap, int rotationDegrees) {
        // Implementation of rotateBitmap method (as in the previous example)
        // ...
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Log.d(TAG, "Permissions not granted by the user.");
                finish();
            }
        }
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onEmptyDetect() {
        runOnUiThread(() -> {
            binding.resultText.setText("No detection");
        });
    }

    @Override
    public void onDetect(List<BoundingBox> boundingBoxes, long inferenceTime) {
        runOnUiThread(() -> {
            if (!boundingBoxes.isEmpty()) {
                // Assuming we're interested in the first detected object
                BoundingBox firstBox = boundingBoxes.get(0);
                binding.resultText.setText(firstBox.getClsName());
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
        if (detector != null) {
            detector.close();
        }
    }

}