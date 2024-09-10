package com.example.asl_recog;

import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

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

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "ASL_Recognition";
    private static final int REQUEST_CODE_PERMISSION = 101;
    private static final String[] REQUIRED_PERMISSION = new String[]{"android.permission.CAMERA"};

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private PreviewView previewView;
    private TextView textView;
    private Hands hands;
    private RandomForestClassifier classifier;
    private List<String> aslClasses;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Log.d(TAG, "onCreate called");

        previewView = findViewById(R.id.cameraView);
        textView = findViewById(R.id.result_text);

        initializeAslClasses();

        if (checkPermissions()) {
            initializeHandsAndCamera();
            initializeRandomForestClassifier();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSION, REQUEST_CODE_PERMISSION);
        }
    }

    private void initializeAslClasses() {
        aslClasses = new ArrayList<>();
        aslClasses.add("0");
        aslClasses.add("1");
        aslClasses.add("2");
        // Add more classes if needed
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
                initializeRandomForestClassifier();
                initializeHandsAndCamera();
            } else {
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    private void initializeRandomForestClassifier() {
        Log.d(TAG, "Starting initializeRandomForestClassifier");
        try {
            // Read the JSON file from the raw folder
            InputStream is = getResources().openRawResource(R.raw.forest_structure);
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
            reader.close();
            is.close();
            String jsonString = sb.toString();
            Log.d(TAG, "JSON string read. Length: " + jsonString.length());

            JSONArray forestArray = new JSONArray(jsonString);
            Log.d(TAG, "Forest JSON parsed. Number of trees: " + forestArray.length());

            List<RandomForestClassifier.DecisionTree> trees = new ArrayList<>();

            for (int i = 0; i < forestArray.length(); i++) {
                Log.d(TAG, "Processing tree " + (i + 1));
                JSONArray treeArray = forestArray.getJSONArray(i);
                RandomForestClassifier.Node root = buildTreeFromJson(treeArray);
                trees.add(new RandomForestClassifier.DecisionTree(root));
            }

            Log.d(TAG, "All trees processed. Creating RandomForestClassifier");
            classifier = new RandomForestClassifier(trees, aslClasses.size());
            Log.i(TAG, "Random Forest Classifier initialized successfully");
        } catch (Resources.NotFoundException e) {
            Log.e(TAG, "forest_structure.json file not found in raw folder: " + e.getMessage());
        } catch (IOException e) {
            Log.e(TAG, "Error reading forest_structure.json: " + e.getMessage());
        } catch (JSONException e) {
            Log.e(TAG, "Error parsing JSON: " + e.getMessage());
        } catch (Exception e) {
            Log.e(TAG, "Error initializing Random Forest Classifier: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private RandomForestClassifier.Node buildTreeFromJson(JSONArray treeArray) throws JSONException {
        Map<Integer, RandomForestClassifier.Node> nodeMap = new HashMap<>();

        for (int i = 0; i < treeArray.length(); i++) {
            JSONObject nodeJson = treeArray.getJSONObject(i);
            int nodeId = nodeJson.getInt("node_id");
            boolean isLeaf = nodeJson.getBoolean("is_leaf");

            if (isLeaf) {
                int prediction = nodeJson.getInt("prediction");
                nodeMap.put(nodeId, new RandomForestClassifier.Node(prediction));
                Log.d(TAG, "Created leaf node " + nodeId + " with prediction " + prediction);
            } else {
                int featureIndex = nodeJson.getInt("feature_index");
                float threshold = (float) nodeJson.getDouble("threshold");
                int leftChild = nodeJson.getInt("left_child");
                int rightChild = nodeJson.getInt("right_child");

                // Create node without children first
                RandomForestClassifier.Node node = new RandomForestClassifier.Node(featureIndex, threshold, null, null);
                nodeMap.put(nodeId, node);
                Log.d(TAG, "Created internal node " + nodeId + " with feature " + featureIndex + " and threshold " + threshold);
            }
        }

        // Second pass to set children for internal nodes
        for (int i = 0; i < treeArray.length(); i++) {
            JSONObject nodeJson = treeArray.getJSONObject(i);
            int nodeId = nodeJson.getInt("node_id");
            boolean isLeaf = nodeJson.getBoolean("is_leaf");

            if (!isLeaf) {
                int leftChild = nodeJson.getInt("left_child");
                int rightChild = nodeJson.getInt("right_child");

                RandomForestClassifier.Node node = nodeMap.get(nodeId);
                node.left = nodeMap.get(leftChild);
                node.right = nodeMap.get(rightChild);

                if (node.left == null || node.right == null) {
                    Log.e(TAG, "Child node not found for node " + nodeId);
                    throw new JSONException("Child node not found for node " + nodeId);
                }
                Log.d(TAG, "Set children for node " + nodeId + ": left = " + leftChild + ", right = " + rightChild);
            }
        }

        RandomForestClassifier.Node root = nodeMap.get(0);
        if (root == null) {
            Log.e(TAG, "Root node (0) not found in the tree");
            throw new JSONException("Root node not found");
        }
        Log.d(TAG, "Tree built successfully. Root node: " + root);
        return root;
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
        if (classifier == null) {
            Log.e(TAG, "Random Forest Classifier is not initialized");
            runOnUiThread(() -> textView.setText("Classifier not initialized"));
            return;
        }

        if (result.multiHandLandmarks().isEmpty()) {
            runOnUiThread(() -> textView.setText("No hands detected"));
            return;
        }

        List<LandmarkProto.NormalizedLandmark> landmarks = result.multiHandLandmarks().get(0).getLandmarkList();

        float[] inputData = new float[21 * 2];  // 21 landmarks, X and Y coordinates
        for (int i = 0; i < landmarks.size(); i++) {
            LandmarkProto.NormalizedLandmark landmark = landmarks.get(i);
            inputData[i * 2] = landmark.getX();
            inputData[i * 2 + 1] = landmark.getY();
        }

        int predictedClass = classifier.predict(inputData);
        String prediction = aslClasses.get(predictedClass);

        runOnUiThread(() -> textView.setText(prediction));
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