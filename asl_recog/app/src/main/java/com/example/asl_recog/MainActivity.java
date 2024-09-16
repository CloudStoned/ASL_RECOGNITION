package com.example.asl_recog;

import android.content.pm.PackageManager;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.ImageProxy;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class MainActivity extends AppCompatActivity implements CameraManager.ImageAnalysisCallback, DataCollector.DataCollectionCallback {
    private static final String TAG = "MainActivity";
    private static final int REQUEST_CODE_PERMISSION = 101;
    private static final String[] REQUIRED_PERMISSION = new String[]{
            "android.permission.CAMERA",
            "android.permission.WRITE_EXTERNAL_STORAGE"
    };

    private PreviewView previewView;
    private TextView textView;
    private CameraManager cameraManager;
    private TorchManager torchManager;
    private DataCollector dataCollector;

    private TextView combinedLettersTextView;
    private Button combineLettersButton;
    private Button clearButton;
    private Button collectDatasetButton;
    private StringBuilder combinedLetters = new StringBuilder();

    private boolean isCollectingData = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.cameraView);
        textView = findViewById(R.id.result_text);
        combinedLettersTextView = findViewById(R.id.combined_letters);
        combineLettersButton = findViewById(R.id.combine_letters);
        clearButton = findViewById(R.id.clear_button);
        collectDatasetButton = findViewById(R.id.collect_dataset_button);

        combineLettersButton.setOnClickListener(v -> combineLetters());
        clearButton.setOnClickListener(v -> clearCombinedLetters());
        collectDatasetButton.setOnClickListener(v -> showDataCollectionDialog());

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


    private void initializeApp() {
        torchManager = new TorchManager(this, textView, combineLettersButton);
        torchManager.loadClasses("classes.txt");
        torchManager.loadModel("2_CLASSES_MODEL.ptl");
        cameraManager = new CameraManager(this, previewView, this);
        cameraManager.startCamera();
        dataCollector = new DataCollector(this);
        dataCollector.setCallback(this);
    }

    // DATA COLLECTION
    private void showDataCollectionDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        View dialogView = getLayoutInflater().inflate(R.layout.data_collection_dialog, null);
        builder.setView(dialogView);

        EditText classNameInput = dialogView.findViewById(R.id.class_name_input);
        EditText datasetSizeInput = dialogView.findViewById(R.id.dataset_size_input);

        builder.setPositiveButton("Start", (dialog, which) -> {
            String className = classNameInput.getText().toString();
            int datasetSize = Integer.parseInt(datasetSizeInput.getText().toString());
            startDataCollection(className, datasetSize);
        });

        builder.setNegativeButton("Cancel", (dialog, which) -> dialog.cancel());

        builder.show();
    }

    private void startDataCollection(String className, int datasetSize) {
        isCollectingData = true;
        dataCollector.startDataCollection(className, datasetSize);
        updateUI();
    }

    private void updateUI() {
        if (isCollectingData) {
            textView.setText(String.format("Collecting: %d / %d",
                    dataCollector.getCurrentImageCount(), dataCollector.getDatasetSize()));
            collectDatasetButton.setEnabled(false);
        } else {
            collectDatasetButton.setEnabled(true);
        }
    }

    @Override
    public void onProgressUpdate(int progress, int total) {
        runOnUiThread(() -> {
            textView.setText(String.format("Collecting: %d / %d", progress, total));
        });
    }

    @Override
    public void onCompleted() {
        runOnUiThread(() -> {
            isCollectingData = false;
            updateUI();
            Toast.makeText(this, "Data collection completed!", Toast.LENGTH_SHORT).show();
        });
    }


    // PYTORCH
    @Override
    public void onImageAnalyzed(@NonNull ImageProxy image) {
        if (isCollectingData) {
            dataCollector.processImage(image);
        } else {
            torchManager.onImageAnalyzed(image);
        }
    }

    private void combineLetters() {
        String currentLetter = textView.getText().toString();
        if(!currentLetter.equals("Result Here")) {
            combinedLetters.append(currentLetter);
            combinedLettersTextView.setText(combinedLetters.toString());
        }
    }

    private void clearCombinedLetters() {
        combinedLetters.setLength(0);
        combinedLettersTextView.setText("Combined Letters");
    }



}