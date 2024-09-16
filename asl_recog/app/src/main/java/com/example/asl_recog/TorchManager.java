package com.example.asl_recog;

import android.annotation.SuppressLint;
import android.content.Context;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageProxy;

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

public class TorchManager {
    private static final String TAG = "TorchManager";
    private final Context context;
    private Module module;
    private List<String> classes;
    private TextView textView;
    private Button combineLettersButton;

    public TorchManager(Context context, TextView textView, Button combineLettersButton) {
        this.context = context;
        this.textView = textView;
        this.combineLettersButton = combineLettersButton;
    }

    public void loadModel(String fileName) {
        File modelFile = new File(context.getFilesDir(), fileName);
        try {
            if (!modelFile.exists()) {
                InputStream inputStream = context.getAssets().open(fileName);
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

    public List<String> loadClasses(String fileName) {
        classes = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(context.getAssets().open(fileName)))) {
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

    @SuppressLint("UnsafeOptInUsageError")
    public void onImageAnalyzed(@NonNull ImageProxy image) {
        int rotation = image.getImageInfo().getRotationDegrees();
        try {
            int cropSize = 224;
            float[] mean = {0.485f, 0.456f, 0.406f};
            float[] std = {0.229f, 0.224f, 0.225f};

            Tensor inputTensor = TensorImageUtils.imageYUV420CenterCropToFloat32Tensor(
                    image.getImage(),
                    rotation,
                    cropSize,
                    cropSize,
                    mean,
                    std
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
                ((android.app.Activity) context).runOnUiThread(() -> {
                    textView.setText(classResult);
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
}