package com.example.asl_recog;

import android.content.Context;
import android.util.Log;
import android.util.Size;

import androidx.annotation.NonNull;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class CameraManager {
    private static final String TAG = "CameraManager";
    private final Context context;
    private final Executor executor = Executors.newSingleThreadExecutor();
    private final ImageAnalysisCallback imageAnalysisCallback;
    private final PreviewView previewView;
    private ProcessCameraProvider cameraProvider;

    public interface ImageAnalysisCallback {
        void onImageAnalyzed(@NonNull ImageProxy image);
    }

    public CameraManager(Context context, PreviewView previewView, ImageAnalysisCallback callback) {
        this.context = context;
        this.previewView = previewView;
        this.imageAnalysisCallback = callback;
    }

    public void startCamera() {
        final ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(context);
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindCameraUseCases();
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Error starting camera", e);
            }
        }, ContextCompat.getMainExecutor(context));
    }

    private void bindCameraUseCases() {
        Preview preview = new Preview.Builder().build();
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(224, 224))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(executor, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy image) {
                imageAnalysisCallback.onImageAnalyzed(image);
            }
        });

        try {
            cameraProvider.unbindAll();
            Camera camera = cameraProvider.bindToLifecycle(
                    (LifecycleOwner) context,
                    cameraSelector,
                    preview,
                    imageAnalysis
            );
        } catch (Exception e) {
            Log.e(TAG, "Use case binding failed", e);
        }
    }
}