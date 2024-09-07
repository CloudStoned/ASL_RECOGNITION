package com.example.asl_recog;

import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.widget.Button;
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
import com.google.mediapipe.solutions.hands.Hands;
import com.google.mediapipe.solutions.hands.HandsOptions;
import com.google.mediapipe.solutions.hands.HandsResult;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
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
    private List<String> CLASSES;
    private final Executor executor = Executors.newSingleThreadExecutor();

    private TextView combinedLettersTextView;
    private Button combineLettersButton;
    private Button clearButton;
    private StringBuilder combinedLetters = new StringBuilder();

    private Hands hands;

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

        // Initialize MediaPipe
        HandsOptions handsOptions = HandsOptions.builder()
                .setStaticImageMode(false)
                .setMaxNumHands(1)
                .setRunOnGpu(true).build();
        hands = new Hands(this, handsOptions);

        // Set up Result Listenetr
        hands.setResultListener(this::analyzeHandsResult);
        hands.setErrorListener(((message, e) ->  Log.e(TAG, "Mediapipe Hands Error: " + message)));

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
        CLASSES = loadClasses("classes.txt");
        loadTorchModule("hand_landmark_model.ptl");
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

    @OptIn(markerClass = ExperimentalGetImage.class)
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

        imageAnalysis.setAnalyzer(executor, image -> {
            Bitmap bitmap = imageToBitmap(image.getImage());
            hands.send(bitmap);
            image.close();
        });

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

    private void analyzeHandsResult(HandsResult handsResult)
    {
        try
        {
            if (handsResult.multiHandLandmarks().isEmpty())
            {
                runOnUiThread(() -> textView.setText("No Hand Detected"));
            }

            // Extract Landmarks
            List<LandmarkProto.NormalizedLandmark> landmarks = handsResult.multiHandLandmarks().get(0).getLandmarkList();
            float[] landmarkArray = convertLandmarksToArray(landmarks);

            // Create Input Tensor
            Tensor inputTensor = Tensor.fromBlob(landmarkArray, new long[]{1, 63});

            // Run Inference
            Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
            float[] scores = outputTensor.getDataAsFloatArray();
            final int maxScoreIdx = argmax(scores);

            if (maxScoreIdx >= 0 && maxScoreIdx < CLASSES.size()) {
                String classResult = CLASSES.get(maxScoreIdx);
                Log.d(TAG, "Detected - " + classResult);
                runOnUiThread(() -> {
                    textView.setText(classResult);
                    // Enable the combine letters button when a letter is detected
                    combineLettersButton.setEnabled(true);
                });
            } else {
                Log.e(TAG, "Index out of bounds for class labels");
            }
        }
        catch (Exception e){
            Log.e(TAG, "Error analyzing image", e);
        }
    }

    private int argmax(float[] array) {
        int maxIdx = 0;
        float maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private float[] convertLandmarksToArray(List<LandmarkProto.NormalizedLandmark> landmarks)
    {
        float[] landmarkArray = new float[landmarks.size() * 3];
        for (int i = 0; i < landmarks.size(); i++) {
            LandmarkProto.NormalizedLandmark landmark = landmarks.get(i);
            landmarkArray[i * 3] = landmark.getX();
            landmarkArray[i * 3 + 1] = landmark.getY();
            landmarkArray[i * 3 + 2] = landmark.getZ();
        }
        return landmarkArray;
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

        YuvImage yuvImage = new YuvImage(nv21, android.graphics.ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }







}