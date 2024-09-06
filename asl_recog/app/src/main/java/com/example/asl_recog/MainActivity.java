package com.example.asl_recog;

import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "TORCH_TORCH";
    private static final int REQUEST_CODE_PERMISSION = 101;
    private static final String[] REQUIRED_PERMISSION = new String[]{"android.permission.CAMERA"};

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private PreviewView previewView;

    private TextView textView;
    private Module module;
    private List<String> classes;
    private final Executor executor = Executors.newSingleThreadExecutor();

    private TextView combinedLettersTextView;
    private Button combineLettersButton;
    private Button clearButton;
    private StringBuilder combinedLetters = new StringBuilder();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.cameraView);
        textView = findViewById(R.id.result_text);
        combinedLettersTextView = findViewById(R.id.combined_letters);
        combineLettersButton = findViewById(R.id.combine_letters);
        clearButton = findViewById(R.id.clear_button);

        combineLettersButton.setOnClickListener(v -> combineLetters());
        clearButton.setOnClickListener(v -> clearCombinedLetters());

        if (checkPermissions()) {
            initializeApp();
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

    private void initializeApp() {
        classes = loadClasses("classes.txt");
        loadTorchModule("2_CLASSES_MODEL.ptl");
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                startCamera(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Error starting camera", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                initializeApp();
            } else {
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    private void startCamera(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder().build();
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(224, 224))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(executor, this::analyzeImage);

        Camera camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
    }

    private void loadTorchModule(String fileName) {
        File modelFile = new File(this.getFilesDir(), fileName);
        try {
            if (!modelFile.exists()) {
                InputStream inputStream = getAssets().open(fileName);
                FileOutputStream outputStream = new FileOutputStream(modelFile);
                byte[] buffer = new byte[2048];
                int bytesRead;
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                }
                inputStream.close();
                outputStream.close();
            }
            module = LiteModuleLoader.load(modelFile.getAbsolutePath());
            Log.d(TAG, "Model successfully loaded");
        } catch (IOException e) {
            Log.e(TAG, "Error loading model", e);
        }
    }

    private void analyzeImage(@NonNull ImageProxy image) {
        int rotation = image.getImageInfo().getRotationDegrees();
        try {
            int cropSize = 224;
            float[] mean = {0.485f, 0.456f, 0.406f};
            float[] std = {0.229f, 0.224f, 0.225f};

            @SuppressLint("UnsafeOptInUsageError")
            Tensor inputTensor = TensorImageUtils.imageYUV420CenterCropToFloat32Tensor(
                    image.getImage(), rotation, cropSize, cropSize,
                    mean, std
            );

            Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();


            float[] scores = outputTensor.getDataAsFloatArray();

            int maxScoreIdx = 0;
            float maxScore = scores[0];
            for (int i = 1; i < scores.length; i++) {
                if (scores[i] > maxScore) {
                    maxScore = scores[i];
                    maxScoreIdx = i;
                }
            }

            if (maxScoreIdx >= 0 && maxScoreIdx < classes.size()) {
                String classResult = classes.get(maxScoreIdx);
                Log.d(TAG, "Detected - " + classResult);
                runOnUiThread(() -> {
                    textView.setText(classResult);
                    // Enable the combine letters button when a letter is detected
                    combineLettersButton.setEnabled(true);
                });
            } else {
                Log.e(TAG, "Index out of bounds for class labels");
            }

        } catch (Exception e) {
            Log.e(TAG, "Error analyzing image", e);
        } finally {
            image.close();
        }
    }

    private List<String> loadClasses(String fileName) {
        List<String> classes = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open(fileName)))) {
            String line;
            while ((line = br.readLine()) != null) {
                classes.add(line);
            }
            Log.d(TAG, "Number of classes loaded: " + classes.size());
        } catch (IOException e) {
            Log.e(TAG, "Error loading classes", e);
        }
        return classes;
    }

    private void combineLetters()
    {
        String currentLetter = textView.getText().toString();
        if(!currentLetter.equals("Result Here"))
        {
            combinedLetters.append(currentLetter);
            combinedLettersTextView.setText(combinedLetters.toString());
        }
    }

    private void clearCombinedLetters()
    {
        combinedLetters.setLength(0);
        combinedLettersTextView.setText("Combined Letters");
    }






}