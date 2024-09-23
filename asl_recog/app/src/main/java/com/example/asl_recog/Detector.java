package com.example.asl_recog;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class Detector {
    private static final float INPUT_MEAN = 0f;
    private static final float INPUT_STANDARD_DEVIATION = 255f;
    private static final DataType INPUT_IMAGE_TYPE = DataType.FLOAT32;
    private static final DataType OUTPUT_IMAGE_TYPE = DataType.FLOAT32;
    private static final float CONFIDENCE_THRESHOLD = 0.3F;
    private static final float IOU_THRESHOLD = 0.5F;
    private static final int NUM_CLASSES = 11; // Number of classes in your model
    private static final int INPUT_SIZE = 640; // Typical YOLOv8 input size

    private final Context context;
    private final String modelPath;
    private final String labelPath;
    private final DetectorListener detectorListener;

    private Interpreter interpreter;
    private List<String> labels = new ArrayList<>();

    private final ImageProcessor imageProcessor;

    public Detector(Context context, String modelPath, String labelPath, DetectorListener detectorListener) {
        this.context = context;
        this.modelPath = modelPath;
        this.labelPath = labelPath;
        this.detectorListener = detectorListener;

        CompatibilityList compatList = new CompatibilityList();

        Interpreter.Options options = new Interpreter.Options();
        if (compatList.isDelegateSupportedOnThisDevice()) {
            GpuDelegateFactory.Options delegateOptions = new GpuDelegateFactory.Options();
            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
            options.addDelegate(gpuDelegate);
        } else {
            options.setNumThreads(4);
        }

        try {
            interpreter = new Interpreter(FileUtil.loadMappedFile(context, modelPath), options);
        } catch (IOException e) {
            e.printStackTrace();
        }

        loadLabels();

        imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
                .add(new CastOp(INPUT_IMAGE_TYPE))
                .build();
    }

    private void loadLabels() {
        try {
            InputStream inputStream = context.getAssets().open(labelPath);
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            String line;
            while ((line = reader.readLine()) != null && !line.isEmpty()) {
                labels.add(line);
            }
            reader.close();
            inputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void restart(boolean isGpu) {
        interpreter.close();

        CompatibilityList compatList = new CompatibilityList();

        Interpreter.Options options = new Interpreter.Options();
        if (isGpu && compatList.isDelegateSupportedOnThisDevice()) {
            GpuDelegateFactory.Options delegateOptions = new GpuDelegateFactory.Options();
            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
            options.addDelegate(gpuDelegate);
        } else {
            options.setNumThreads(4);
        }

        try {
            interpreter = new Interpreter(FileUtil.loadMappedFile(context, modelPath), options);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void close() {
        interpreter.close();
    }


    public void detect(Bitmap frame) {
        long inferenceTime = SystemClock.uptimeMillis();

        TensorImage tensorImage = TensorImage.fromBitmap(frame);
        tensorImage = imageProcessor.process(tensorImage);

        TensorBuffer outputBuffer = TensorBuffer.createFixedSize(new int[]{1, 8400, NUM_CLASSES + 4}, OUTPUT_IMAGE_TYPE);
        interpreter.run(tensorImage.getBuffer(), outputBuffer.getBuffer());

        List<BoundingBox> bestBoxes = processModelOutput(outputBuffer.getFloatArray(), frame.getWidth(), frame.getHeight());
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime;

        if (bestBoxes.isEmpty()) {
            detectorListener.onEmptyDetect();
        } else {
            detectorListener.onDetect(bestBoxes, inferenceTime);
        }
    }

    private List<BoundingBox> processModelOutput(float[] output, int sourceWidth, int sourceHeight) {
        List<BoundingBox> boundingBoxes = new ArrayList<>();

        for (int i = 0; i < 8400; i++) {
            int offset = i * (NUM_CLASSES + 4);

            float[] scores = new float[NUM_CLASSES];
            System.arraycopy(output, offset + 4, scores, 0, NUM_CLASSES);

            int classIndex = 0;
            float maxScore = 0;
            for (int j = 0; j < NUM_CLASSES; j++) {
                if (scores[j] > maxScore) {
                    maxScore = scores[j];
                    classIndex = j;
                }
            }

            if (maxScore > CONFIDENCE_THRESHOLD) {
                float x = output[offset];
                float y = output[offset + 1];
                float w = output[offset + 2];
                float h = output[offset + 3];

                float x1 = (x - w/2) * sourceWidth;
                float y1 = (y - h/2) * sourceHeight;
                float x2 = (x + w/2) * sourceWidth;
                float y2 = (y + h/2) * sourceHeight;

                BoundingBox box = new BoundingBox(
                        x1, y1, x2, y2,
                        x, y, w, h,
                        maxScore, classIndex, labels.get(classIndex)
                );
                boundingBoxes.add(box);
            }
        }

        return applyNMS(boundingBoxes);
    }

    private List<BoundingBox> applyNMS(List<BoundingBox> boxes) {
        boxes.sort((b1, b2) -> Float.compare(b2.getCnf(), b1.getCnf()));
        List<BoundingBox> selectedBoxes = new ArrayList<>();

        while (!boxes.isEmpty()) {
            BoundingBox first = boxes.remove(0);
            selectedBoxes.add(first);

            boxes.removeIf(box -> calculateIoU(first, box) >= IOU_THRESHOLD);
        }

        return selectedBoxes;
    }

    private float calculateIoU(BoundingBox box1, BoundingBox box2) {
        float x1 = Math.max(box1.getX1(), box2.getX1());
        float y1 = Math.max(box1.getY1(), box2.getY1());
        float x2 = Math.min(box1.getX2(), box2.getX2());
        float y2 = Math.min(box1.getY2(), box2.getY2());
        float intersectionArea = Math.max(0F, x2 - x1) * Math.max(0F, y2 - y1);
        float box1Area = box1.getW() * box1.getH();
        float box2Area = box2.getW() * box2.getH();
        return intersectionArea / (box1Area + box2Area - intersectionArea);
    }

    public interface DetectorListener {
        void onEmptyDetect();
        void onDetect(List<BoundingBox> boundingBoxes, long inferenceTime);
    }


}
