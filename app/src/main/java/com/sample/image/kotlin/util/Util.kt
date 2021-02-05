package com.sample.image.kotlin.util

import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.sample.image.kotlin.view.HomeActivity
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.ArrayList

class Util {
    companion object {
        public fun isPermissionGranted(context: Context, permission: String): Boolean {
            if (ContextCompat.checkSelfPermission(context, permission)
                == PackageManager.PERMISSION_GRANTED
            ) {
                return true
            }
            return false
        }

        public fun calculate_distance(
            ori_embedding: Array<FloatArray>,
            test_embedding: Array<FloatArray>
        ): Double {
            var sum = 0.0
            for (i in 0..127) {
                sum = sum + Math.pow(
                    (ori_embedding[0][i] - test_embedding[0][i]).toDouble(), 2.0
                )
            }
            return Math.sqrt(sum)
        }

        public fun getRequiredPermissions(activity: AppCompatActivity): Array<String?> {
            return try {
                val info = activity.packageManager
                    .getPackageInfo(activity.packageName, PackageManager.GET_PERMISSIONS)
                val ps = info.requestedPermissions
                if (ps != null && ps.isNotEmpty()) {
                    ps
                } else {
                    arrayOfNulls(0)
                }
            } catch (e: Exception) {
                arrayOfNulls(0)
            }
        }

        public fun getRuntimePermissions(activity: AppCompatActivity, requestCode: Int) {
            val allNeededPermissions = ArrayList<String>()
            for (permission in getRequiredPermissions(activity)) {
                permission?.let {
                    if (!isPermissionGranted(activity, it)) {
                        allNeededPermissions.add(permission)
                    }
                }
            }

            if (allNeededPermissions.isNotEmpty()) {
                ActivityCompat.requestPermissions(
                    activity, allNeededPermissions.toTypedArray(), requestCode
                )
            }
        }

        public fun allPermissionsGranted(activity: AppCompatActivity): Boolean {
            for (permission in Util.getRequiredPermissions(activity)) {
                permission?.let {
                    if (!Util.isPermissionGranted(activity, it)) {
                        return false
                    }
                }
            }
            return true
        }

        public fun loadmodelfile(activity: Activity): MappedByteBuffer? {
            val fileDescriptor = activity.assets.openFd("Qfacenet.tflite")
            val inputStream =
                FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startoffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            return fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                startoffset,
                declaredLength
            )
        }

        public fun getPreprocessNormalizeOp(): TensorOperator? {
            return NormalizeOp(
                IMAGE_MEAN,
                IMAGE_STD
            )
        }

        private val IMAGE_MEAN = 0.0f
        private val IMAGE_STD = 1.0f



        public fun loadImage(
            bitmap: Bitmap,
            inputImageBuffer: TensorImage,
            targeHeight: Int,
            targetWidth: Int
        ): TensorImage {
            // Loads bitmap into a TensorImage.
            inputImageBuffer.load(bitmap)

            // Creates processor for the TensorImage.
            val cropSize = Math.min(bitmap.width, bitmap.height)
            // TODO(b/143564309): Fuse ops inside ImageProcessor.
            val imageProcessor: ImageProcessor = ImageProcessor.Builder()
                .add(ResizeWithCropOrPadOp(cropSize, cropSize))
                .add(ResizeOp(targeHeight, targetWidth, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .add(Util.getPreprocessNormalizeOp())
                .build()
            return imageProcessor.process(inputImageBuffer)
        }
    }


}