package com.sakhiya.tts

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Sakhiya GGML C++ engine for Android (Nepali + Sanskrit TTS, 24 kHz mono float32 PCM).
 *
 * Build: add `cpp/CMakeLists.txt` as an externalNativeBuild (target `sakhiya_jni`, no third-party dependency), ship a
 * `.gguf` (e.g. `sakhiya_atom_lite_q8_0.gguf`, ~3.5 MB) in assets or download it from
 * https://huggingface.co/NikitHamal/sakhiya-gguf, then:
 *
 *     val tts = SakhiyaGGML.fromFile(File(filesDir, "sakhiya_atom_lite_q8_0.gguf"), threads = 2)
 *     tts.synthesize("नमस्ते नेपाल", style = "news").use { it.play() }   // zero-copy FloatBuffer view of native memory
 */
class SakhiyaGGML private constructor(private var handle: Long) : AutoCloseable {

    val tier: String get() = nativeTier(handle)

    /** Zero-copy PCM: [floats] is a direct FloatBuffer aliasing native memory; [close] frees it. */
    class Pcm internal constructor(private var bb: ByteBuffer?) : AutoCloseable {
        val floats: FloatBuffer get() = bb!!.order(ByteOrder.nativeOrder()).asFloatBuffer()
        val sampleRate: Int get() = SAMPLE_RATE
        val seconds: Float get() = (bb!!.capacity() / 4) / SAMPLE_RATE.toFloat()
        override fun close() { bb?.let { nativeFreePcm(it) }; bb = null }

        fun play() {
            val n = bb!!.capacity() / 4
            val track = AudioTrack.Builder()
                .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(SAMPLE_RATE).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                .setBufferSizeInBytes(maxOf(n * 4, 4)).setTransferMode(AudioTrack.MODE_STATIC).build()
            track.write(floats, n, AudioTrack.WRITE_BLOCKING)
            track.play()
        }
    }

    /** Synthesizes [text]. Params <= 0 (pitchShift <= -900) fall back to the style defaults. Close the returned [Pcm]. */
    @JvmOverloads
    fun synthesize(
        text: String, lang: String = "ne", style: String = "news",
        rate: Float = -1f, pitchShift: Float = -999f, pitchScale: Float = -1f,
        pauseScale: Float = -1f, energyScale: Float = -1f, svara: Float = 1f,
    ): Pcm = Pcm(
        nativeSynthesize(handle, text, lang, style, rate, pitchShift, pitchScale, pauseScale, energyScale, svara)
            ?: error("sakhiya: synthesis failed")
    )

    override fun close() { if (handle != 0L) { nativeFree(handle); handle = 0L } }

    companion object {
        const val SAMPLE_RATE = 24000

        init { System.loadLibrary("sakhiya_jni") }

        fun fromFile(file: File, threads: Int = 2) = SakhiyaGGML(nativeInitFile(file.absolutePath, threads))

        /** [gguf] must be a *direct* ByteBuffer (e.g. a memory-mapped asset). The engine copies what it needs. */
        fun fromBuffer(gguf: ByteBuffer, threads: Int = 2) = SakhiyaGGML(nativeInitBuffer(gguf, threads))

        fun fromAsset(ctx: Context, name: String, threads: Int = 2): SakhiyaGGML {
            val bytes = ctx.assets.open(name).use { it.readBytes() }
            val direct = ByteBuffer.allocateDirect(bytes.size).put(bytes).also { it.flip() }
            return fromBuffer(direct, threads)
        }

        @JvmStatic private external fun nativeInitFile(path: String, threads: Int): Long
        @JvmStatic private external fun nativeInitBuffer(buf: ByteBuffer, threads: Int): Long
        @JvmStatic private external fun nativeFree(handle: Long)
        @JvmStatic private external fun nativeTier(handle: Long): String
        @JvmStatic private external fun nativeSynthesize(
            handle: Long, text: String, lang: String, style: String,
            rate: Float, pitchShift: Float, pitchScale: Float, pauseScale: Float, energyScale: Float, svara: Float,
        ): ByteBuffer?
        @JvmStatic internal external fun nativeFreePcm(buf: ByteBuffer)
    }
}
