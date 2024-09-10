package com.example.asl_recog;

import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
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


public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MediaPipeHandDetection";
    private static final int REQUEST_CODE_PERMISSION = 101;
    private static final String[] REQUIRED_PERMISSION = new String[]{"android.permission.CAMERA"};

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private PreviewView previewView;
    private TextView textView;
    private ImageView overlayView;

    // Mediapipe
    private Hands hands;
    private Paint boxPaint;
    private Paint landmarkPaint;

    // Pytorch
    private Module module;
    private List<String> classes;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.cameraView);
        textView = findViewById(R.id.result_text);
        overlayView = findViewById(R.id.overlayView);

        // Initialize paints
        boxPaint = new Paint();
        boxPaint.setColor(Color.RED);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(5f);

        landmarkPaint = new Paint();
        landmarkPaint.setColor(Color.GREEN);
        landmarkPaint.setStyle(Paint.Style.FILL);
        landmarkPaint.setStrokeWidth(5f);


        if (checkPermissions()) {
            initializeHandsAndCamera();
            classes = loadClasses("classes.txt");
            loadTorchModule("hand_landmark_model.pt");
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSION, REQUEST_CODE_PERMISSION);
        }
    }

    // PYTORCH
    private void loadTorchModule(String fileName) {
        File modelFile = new File(this.getFilesDir(), fileName);
        try {
            if (!modelFile.exists()) {
                InputStream inputStream = getAssets().open(fileName);
                FileOutputStream outputStream = new FileOutputStream(modelFile);
                byte[] buffer = new byte[4 * 1024];
                int bytesRead;
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                }
                inputStream.close();
                outputStream.close();
            }
            module = (Module) LiteModuleLoader.load(modelFile.getAbsolutePath());
            Log.d(TAG, "Model successfully loaded");
        } catch (IOException e) {
            Log.e(TAG, "Error loading model", e);
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

    // CAMERA
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

    private void initializeHandsAndCamera() {
        HandsOptions handsOptions = HandsOptions.builder()
                .setStaticImageMode(false)
                .setMaxNumHands(1)
                .setMinDetectionConfidence(0.5f)
                .build();
        hands = new Hands(this, handsOptions);

        hands.setErrorListener((message, e) -> Log.e(TAG, "MediaPipe Hands error: " + message));

        startCamera();
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

    // INFERENCING
    private void processHandsResult(HandsResult result) {
        runOnUiThread(() -> {
            if (result.multiHandLandmarks().isEmpty())
            {
                textView.setText("No hands detected");
                overlayView.setImageBitmap(null);
            }

            else
            {
                List<LandmarkProto.NormalizedLandmark> landmarks = result.multiHandLandmarks().get(0).getLandmarkList();
                float[] landmarkArray = landmarksToArray(landmarks);
                int predictedClass = runInference(landmarkArray);
                String prediction = classes.get(predictedClass);

                textView.setText("Predicted sign: " + prediction);

                Bitmap outputBitmap = Bitmap.createBitmap(previewView.getWidth(), previewView.getHeight(), Bitmap.Config.ARGB_8888);
                Canvas canvas = new Canvas(outputBitmap);

                drawLandmarks(canvas, landmarks);
                drawBoundingBox(canvas, landmarks);

                overlayView.setImageBitmap(outputBitmap);
            }
        });
    }

    private int runInference(float[] landmarkArray) {
        Tensor inputTensor = preprocess(landmarkArray);
        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        float[] scores = outputTensor.getDataAsFloatArray();
        return argmax(scores);
    }


    // TOOLS
    private float[] landmarksToArray(List<LandmarkProto.NormalizedLandmark> landmarks) {
        float[] landmarkArray = new float[landmarks.size() * 2];
        for (int i = 0; i < landmarks.size(); i++) {
            LandmarkProto.NormalizedLandmark landmark = landmarks.get(i);
            landmarkArray[i * 2] = landmark.getX();
            landmarkArray[i * 2 + 1] = landmark.getY();
        }
        return landmarkArray;
    }

    private Tensor preprocess(float[] landmarkArray) {
        float[] normalizedData = new float[landmarkArray.length];
        for (int i = 0; i < landmarkArray.length; i++) {
            normalizedData[i] = (landmarkArray[i] - 0.5f) * 2.0f;
        }
        return Tensor.fromBlob(normalizedData, new long[]{1, landmarkArray.length});
    }

    private int argmax(float[] array) {
        int maxIdx = 0;
        float maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxIdx = i;
                maxVal = array[i];
            }
        }
        return maxIdx;
    }

    private void drawLandmarks(Canvas canvas, List<LandmarkProto.NormalizedLandmark> landmarks) {
        for (LandmarkProto.NormalizedLandmark landmark : landmarks) {
            float x = landmark.getX() * canvas.getWidth();
            float y = landmark.getY() * canvas.getHeight();
            canvas.drawCircle(x, y, 8f, landmarkPaint);
        }
    }

    private void drawBoundingBox(Canvas canvas, List<LandmarkProto.NormalizedLandmark> landmarks) {
        RectF boundingBox = getBoundingBox(landmarks);
        canvas.drawRect(boundingBox, boxPaint);
    }

    private RectF getBoundingBox(List<LandmarkProto.NormalizedLandmark> landmarks) {
        float minX = Float.MAX_VALUE;
        float minY = Float.MAX_VALUE;
        float maxX = Float.MIN_VALUE;
        float maxY = Float.MIN_VALUE;

        for (LandmarkProto.NormalizedLandmark landmark : landmarks) {
            minX = Math.min(minX, landmark.getX());
            minY = Math.min(minY, landmark.getY());
            maxX = Math.max(maxX, landmark.getX());
            maxY = Math.max(maxY, landmark.getY());
        }

        int width = previewView.getWidth();
        int height = previewView.getHeight();
        return new RectF(minX * width, minY * height, maxX * width, maxY * height);
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