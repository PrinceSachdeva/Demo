package com.sample.image.kotlin

import android.graphics.Bitmap
import com.google.mlkit.vision.face.Face

public interface OnFaceSelectedListener {
    fun onFaceSelected(face: Face, bitmap: Bitmap)

    fun onFaceSelectionError()
}