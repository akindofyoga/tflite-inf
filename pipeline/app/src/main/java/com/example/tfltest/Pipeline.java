package com.example.tfltest;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.os.BatteryManager;
import android.os.Environment;
import android.os.SystemClock;

import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.task.core.BaseOptions;
import org.tensorflow.lite.task.vision.classifier.Classifications;
import org.tensorflow.lite.task.vision.classifier.ImageClassifier;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;
import org.tensorflow.lite.gpu.CompatibilityList;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

public class Pipeline {

    ImageClassifier imageClassifier = null;
    ObjectDetector objectDetector = null;
    final private String odname;

    public Pipeline(String odname) {
        this.odname = odname;

        try {
            ImageClassifier.ImageClassifierOptions.Builder optionsBuilder =
                    ImageClassifier.ImageClassifierOptions.builder()
                            .setScoreThreshold(0.4f)
                            .setMaxResults(1);

            BaseOptions.Builder baseOptionsBuilder = BaseOptions.builder().setNumThreads(1);
            
            // Works on Google Glass even though 
            // (new CompatibilityList()).isDelegateSupportedOnThisDevice() returns false.
            //baseOptionsBuilder.useGpu();

            optionsBuilder.setBaseOptions(baseOptionsBuilder.build());
            File modelFile = new File("/sdcard/models/stirling/r50.tflite");

            imageClassifier = ImageClassifier.createFromFileAndOptions(
                    modelFile, optionsBuilder.build());
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            ObjectDetector.ObjectDetectorOptions.Builder optionsBuilder =
                    ObjectDetector.ObjectDetectorOptions.builder()
                            .setScoreThreshold(0.4f)
                            .setMaxResults(1);

            BaseOptions.Builder baseOptionsBuilder = BaseOptions.builder().setNumThreads(1);

            // This check is necessary because baseOptionsBuilder.useGpu(); will not work on Google
            // glass.
            //if ((new CompatibilityList()).isDelegateSupportedOnThisDevice()) {
                //baseOptionsBuilder.useGpu();
            //}
            
            optionsBuilder.setBaseOptions(baseOptionsBuilder.build());

            File modelFile = new File("/sdcard/models/stirling/" + odname);
            objectDetector = ObjectDetector.createFromFileAndOptions(
                    modelFile, optionsBuilder.build());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void runTest(BatteryManager batteryManager) {
        int good = 0;
        int bad = 0;

        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream("/sdcard/output/stirling/" + odname + "_" + System.currentTimeMillis() + ".txt");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        classifyImage("/sdcard/test_images/stirling/1screw/2_frame-0000.jpg", "warmup");
        System.out.println("Finished warmup");

        File testImages = new File( Environment.getExternalStorageDirectory().getPath()
                + "/test_images/stirling");
        int count = 0;
        long start = SystemClock.uptimeMillis();
        long startEnergy = batteryManager.getLongProperty(
                BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER);

        int total = 0;

        for (File classDir : testImages.listFiles()) {
            String correctClass = classDir.getName();
            for (File imageFile : classDir.listFiles()) {
                if (classifyImage(imageFile.getPath(), correctClass)) {
                    good++;
                } else {
                    bad++;
                }

                if (count == 20) {
                    long end = SystemClock.uptimeMillis();
                    System.out.println("time change: " + (end - start) + "ms");

                    long endEnergy = batteryManager.getLongProperty(
                            BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER);
                    System.out.println("energy: " + endEnergy);
                    System.out.println("energy change: " + (endEnergy - startEnergy) + "nWh");

                    try {
                        fos.write(("time change: " + (end - start) + "ms energy change: " + (startEnergy - endEnergy) + " nWh\n").getBytes());
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    System.out.println("Good: " + good + " Bad: " + bad);
                    count = 0;
                    start = SystemClock.uptimeMillis();

                    startEnergy = batteryManager.getLongProperty(
                            BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER);
                }

                if (total == 120) {
                    try {
                        fos.write(("Good: " + good + " Bad: " + bad + "\n").getBytes());
                        fos.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    System.out.println("Good: " + good + " Bad: " + bad);
                    return;
                }
                total++;
                count++;
            }
        }
    }

    private boolean classifyImage(String imagePath, String correctClass) {
        Bitmap image = BitmapFactory.decodeFile(imagePath);
        ImageProcessor imageProcessor = (new ImageProcessor.Builder()).build();
        TensorImage tensorImage = imageProcessor.process(TensorImage.fromBitmap(image));

        List<Detection> detections = objectDetector.detect(tensorImage);
        if (detections.size() == 0) {
            return false;
        }
        if (detections.size() > 1) {
            throw new RuntimeException();
        }
        RectF rectF = detections.get(0).getBoundingBox();

        if ((rectF.top < 0) || (rectF.left < 0) || (rectF.width() == 0) ||
                (rectF.height() == 0)) {
            return false;
        }

        if ((rectF.left + rectF.width()) > image.getWidth()) {
            System.out.println("bad");
            return false;
        }
//        List<Category> dets = detections.get(0).getCategories();
//
//        if (dets.size() == 0) {
//            return false;
//        }
//
//        return dets.get(0).getLabel().equals(correctClass);

        image = Bitmap.createBitmap(image, (int)rectF.left, (int)rectF.top,
                (int)rectF.width(), (int)rectF.height());

        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        image.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
        ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(
                byteArrayOutputStream.toByteArray());
        image = BitmapFactory.decodeStream(byteArrayInputStream);

        imageProcessor = (new ImageProcessor.Builder()).build();
        tensorImage = imageProcessor.process(TensorImage.fromBitmap(image));

        List<Classifications> results = imageClassifier.classify(tensorImage);
        if ((results.size() != 1) || (results.get(0).getCategories().size() == 0)) {
            return false;
        }

        return results.get(0).getCategories().get(0).getLabel().equals(correctClass);
    }
}
