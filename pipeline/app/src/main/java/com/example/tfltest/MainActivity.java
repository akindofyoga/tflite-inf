package com.example.tfltest;

import static android.provider.Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION;

import android.Manifest;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.net.Uri;
import android.os.BatteryManager;
import android.os.Bundle;

import com.google.android.material.snackbar.Snackbar;

import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import android.os.Environment;
import android.os.SystemClock;
import android.provider.Settings;
import android.view.View;

import androidx.core.content.ContextCompat;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import com.example.tfltest.databinding.ActivityMainBinding;

import android.view.Menu;
import android.view.MenuItem;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.core.BaseOptions;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.task.vision.classifier.Classifications;
import org.tensorflow.lite.task.vision.classifier.ImageClassifier;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;

public class MainActivity extends AppCompatActivity {

    private AppBarConfiguration appBarConfiguration;
    private ActivityMainBinding binding;

    private ExecutorService pool;

    private volatile boolean running;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setSupportActionBar(binding.toolbar);

        NavController navController = Navigation.findNavController(this,
                R.id.nav_host_fragment_content_main);
        appBarConfiguration = new AppBarConfiguration.Builder(navController.getGraph()).build();
        NavigationUI.setupActionBarWithNavController(this, navController, appBarConfiguration);

        binding.fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
                        .setAction("Action", null).show();
            }
        });

        // Request ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION on Vuzix Blade 2
//        if (!Environment.isExternalStorageManager()) {
//            Intent intent = new Intent(ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
//                    Uri.parse("package:" + BuildConfig.APPLICATION_ID));
//            startActivity(intent);
//        }

        // Request READ_EXTERNAL_STORAGE on ODG and Google Glass
        int permission1 = ContextCompat.checkSelfPermission(
                this, Manifest.permission.READ_EXTERNAL_STORAGE);
        int permission2 = ContextCompat.checkSelfPermission(
                this, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        if ((permission1 != PackageManager.PERMISSION_GRANTED) ||
                (permission2 != PackageManager.PERMISSION_GRANTED)) {
            requestPermissions(new String[] {Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE}, 0);
        }

        pool = Executors.newFixedThreadPool(2);
        pool.execute(() -> {
            BatteryManager batteryManager =
                    (BatteryManager) getSystemService(Context.BATTERY_SERVICE);

            try {
                FileOutputStream fos = new FileOutputStream(
                        "/sdcard/output/bg_thread/baseline_" +
                                System.currentTimeMillis() + ".txt");
                BatteryReceiver batteryReceiver = new BatteryReceiver();
                IntentFilter intentFilter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
                registerReceiver(batteryReceiver, intentFilter);

                for (int i = 0; i < 3000; i++) {
                    long start = SystemClock.uptimeMillis();

                    long toWait = Math.max(0, ((start + 100) - SystemClock.uptimeMillis()));
                    Thread.sleep(toWait);

                    int current = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW);
                    int voltage = batteryReceiver.getVoltage();
                    fos.write(("current: " + current + " voltage: " + voltage + "\n").getBytes());
                }
                fos.close();
                unregisterReceiver(batteryReceiver);
            } catch (InterruptedException | IOException e) {
                e.printStackTrace();
            }

            for (String odname : new String[] {"ed0.tflite", "ed1.tflite", "ed2.tflite"}) {
                running = true;
                pool.execute(() -> {
                    System.out.println("starting");
                    Pipeline pipeline = new Pipeline(odname);
                    pipeline.runTest(batteryManager);
                    running = false;
                });

                measureBattery(odname, batteryManager);
            }
            finish();
        });
    }

    private void measureBattery(String odname, BatteryManager batteryManager) {
        try {
            FileOutputStream fos = new FileOutputStream(
                    "/sdcard/output/bg_thread/" + odname + "_" +
                            System.currentTimeMillis() + ".txt");
            BatteryReceiver batteryReceiver = new BatteryReceiver();
            IntentFilter intentFilter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
            registerReceiver(batteryReceiver, intentFilter);

            while (true) {
                long start = SystemClock.uptimeMillis();

                long toWait = Math.max(0, ((start + 100) - SystemClock.uptimeMillis()));
                Thread.sleep(toWait);

                if (!running) {
                    break;
                }
                int current = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW);
                int voltage = batteryReceiver.getVoltage();
                fos.write(("current: " + current + " voltage: " + voltage + "\n").getBytes());
            }
            fos.close();
            unregisterReceiver(batteryReceiver);
        } catch (InterruptedException | IOException e) {
            e.printStackTrace();
        }
    }

    private class BatteryReceiver extends BroadcastReceiver {
        private int voltage;

        private int getVoltage() {
            return voltage;
        }

        @Override
        public void onReceive(Context context, Intent intent) {
            voltage = intent.getIntExtra(BatteryManager.EXTRA_VOLTAGE, Integer.MIN_VALUE);
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    public boolean onSupportNavigateUp() {
        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_content_main);
        return NavigationUI.navigateUp(navController, appBarConfiguration)
                || super.onSupportNavigateUp();
    }
}