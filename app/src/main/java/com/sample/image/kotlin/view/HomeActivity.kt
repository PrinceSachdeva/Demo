package com.sample.image.kotlin.view

import android.app.Activity
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.util.Pair
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.google.mlkit.vision.face.Face
import com.sample.image.R
import com.sample.image.kotlin.BitmapUtils
import com.sample.image.kotlin.GraphicOverlay
import com.sample.image.kotlin.OnFaceSelectedListener
import com.sample.image.kotlin.VisionImageProcessor
import com.sample.image.kotlin.facedetector.FaceDetectorProcessor
import com.sample.image.kotlin.util.Util
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.math.max

class HomeActivity : AppCompatActivity(), ActivityCompat.OnRequestPermissionsResultCallback {
    private var imageSizeX: Int = 0
    private var imageSizeY: Int = 0
    private lateinit var tflite: Interpreter
    private var imageProcessor: VisionImageProcessor? = null
    private var imageUri: Uri? = null
    private var preview: ImageView? = null
    private var graphicOverlay: GraphicOverlay? = null
    private var isFirstImageSelected = false;
    private val PERMISSION_REQUESTS = 1
    var ori_embedding =
        Array(1) { FloatArray(128) }
    var test_embedding =
        Array(1) { FloatArray(128) }
    private val targetedWidthHeight: Pair<Int, Int>
        get() {
            val targetWidth: Int = 1440
            val targetHeight: Int = 2246

            return Pair(targetWidth, targetHeight)
        }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_home)

        preview = findViewById(R.id.preview)
        graphicOverlay = findViewById(R.id.graphic_overlay)

        findViewById<Button>(R.id.select_image_button).setOnClickListener(clickListener)
        findViewById<Button>(R.id.select_image_button_1).setOnClickListener(clickListener)
        findViewById<Button>(R.id.verify_image).setOnClickListener(clickListener)


        if (!Util.allPermissionsGranted(this))
            Util.getRuntimePermissions(this, PERMISSION_REQUESTS)

        tflite = Interpreter(Util.loadmodelfile(this)!!)
    }


    override fun onResume() {
        super.onResume()
        imageProcessor =
            FaceDetectorProcessor(
                this,
                null,
                getOnFaceSelectedListener()
            )
    }


    override fun onActivityResult(
        requestCode: Int,
        resultCode: Int,
        data: Intent?
    ) {
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == Activity.RESULT_OK) {
            tryReloadAndDetectInImage()
        } else {
            super.onActivityResult(requestCode, resultCode, data)
        }
    }

    private fun startCameraIntentForResult() {
        if (!Util.allPermissionsGranted(this)) {
            Util.getRuntimePermissions(this, PERMISSION_REQUESTS)
            return
        }

        // Clean up last time's image
        imageUri = null
        preview!!.setImageBitmap(null)
        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        if (takePictureIntent.resolveActivity(packageManager) != null) {
            val file = File(applicationContext.getCacheDir(), "image.png")
            imageUri = FileProvider.getUriForFile(
                applicationContext,
                "com.sample.image.fileprovider",
                file
            )
            takePictureIntent.setFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri)
            startActivityForResult(
                takePictureIntent,
                REQUEST_IMAGE_CAPTURE
            )
        }
    }

    private fun tryReloadAndDetectInImage() {
        try {
            if (imageUri == null) {
                return
            }
            val imageBitmap = BitmapUtils.getBitmapFromContentUri(contentResolver, imageUri)
                ?: return
            // Clear the overlay first
            graphicOverlay!!.clear()
            // Get the dimensions of the image view
            val targetedSize = targetedWidthHeight
            // Determine how much to scale down the image
            val scaleFactor = max(
                imageBitmap.width.toFloat() / targetedSize.first.toFloat(),
                imageBitmap.height.toFloat() / targetedSize.second.toFloat()
            )
            val resizedBitmap = Bitmap.createScaledBitmap(
                imageBitmap,
                (imageBitmap.width / scaleFactor).toInt(),
                (imageBitmap.height / scaleFactor).toInt(),
                true
            )
            preview!!.setImageBitmap(resizedBitmap)
            if (imageProcessor != null) {
                graphicOverlay!!.setImageSourceInfo(
                    resizedBitmap.width, resizedBitmap.height, /* isFlipped= */false
                )
                imageProcessor!!.processBitmap(resizedBitmap, graphicOverlay)
            } else {
                Log.e(
                    TAG,
                    "Null imageProcessor, please check adb logs for imageProcessor creation error"
                )
            }
        } catch (e: Exception) {
            Log.e(
                TAG,
                "Error retrieving saved image"
            )
            imageUri = null
        }
    }


    companion object {
        private const val TAG = "StillImageActivity"
        private const val REQUEST_IMAGE_CAPTURE = 1001
    }

    fun get_embaddings(bitmap: Bitmap?) {
        var inputImageBuffer: TensorImage
        var embedding =
            Array(1) { FloatArray(128) }
        if (isFirstImageSelected)
            embedding = ori_embedding
        else
            embedding = test_embedding
        val imageTensorIndex = 0
        val imageShape: IntArray =
            tflite.getInputTensor(imageTensorIndex).shape() // {1, height, width, 3}
        imageSizeY = imageShape[1]
        imageSizeX = imageShape[2]
        val imageDataType: DataType =
            tflite.getInputTensor(imageTensorIndex).dataType()
        inputImageBuffer = TensorImage(imageDataType)
        inputImageBuffer = Util.loadImage(bitmap!!, inputImageBuffer, imageSizeX, imageSizeY)
        tflite.run(inputImageBuffer.buffer, embedding)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>, grantResults: IntArray
    ) {
        when (requestCode) {
            PERMISSION_REQUESTS -> {

                if (grantResults.isEmpty() || grantResults[0] != PackageManager.PERMISSION_GRANTED) {

                } else {
                    startCameraIntentForResult()
                }
            }
        }
    }

    val clickListener = View.OnClickListener { view ->

        when (view.getId()) {
            R.id.verify_image -> {
                verify()
            }
            R.id.select_image_button -> {
                isFirstImageSelected = true
                startCameraIntentForResult()
            }
            R.id.select_image_button_1 -> {
                isFirstImageSelected = false
                startCameraIntentForResult()
            }
        }
    }

    fun getOnFaceSelectedListener(): OnFaceSelectedListener {
        return object : OnFaceSelectedListener {
            override fun onFaceSelected(face: Face, bitmap: Bitmap) {
                val bounds = face.boundingBox
                var cropped: Bitmap? = null;
                try {
                    cropped = Bitmap.createBitmap(
                        bitmap,
                        bounds.left,
                        bounds.top,
                        bounds.width(),
                        bounds.height()
                    )
                } catch (e: java.lang.Exception) {
                    onFaceSelectionErrorFromFireBase()
                }
                cropped?.let {
                    get_embaddings(it)
                }
            }

            override fun onFaceSelectionError() {
                onFaceSelectionErrorFromFireBase()
            }

        }
    }

    private fun onFaceSelectionErrorFromFireBase() {
        applicationContext.showToast("Face Detection Error!!! Please Try Again")
    }

    fun verify() {
        if (ori_embedding.get(0).distinct().size != 1 && test_embedding.get(0)
                .distinct().size != 1
        ) {
            val distance = Util.calculate_distance(ori_embedding, test_embedding)
            if (distance < 6.0)
                applicationContext.showToast("Same Faces");
            else
                applicationContext.showToast("Different Faces");

        } else {
            onFaceSelectionErrorFromFireBase()
        }
    }

}

private fun Context.showToast(message: String) {
    Toast.makeText(this, message, Toast.LENGTH_LONG).show()
}
