package com.yourname.aiassistant

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.yourname.aiassistant.data.llm.LlamaEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val engine: LlamaEngine? = try {
            LlamaEngine()
        } catch (e: UnsatisfiedLinkError) {
            null
        }

        setContent {
            var status by remember {
                mutableStateOf(
                    if (engine == null) "Native library failed to load"
                    else "Tap below, then choose your GGUF model file"
                )
            }
            val scope = rememberCoroutineScope()

            val pickFile = rememberLauncherForActivityResult(
                contract = ActivityResultContracts.OpenDocument()
            ) { uri: Uri? ->
                if (engine == null) return@rememberLauncherForActivityResult
                if (uri == null) {
                    status = "No file selected"
                    return@rememberLauncherForActivityResult
                }

                scope.launch {
                    status = "Copying model into the app..."

                    val modelFile = File(filesDir, "model.gguf")

                    val copied = withContext(Dispatchers.IO) {
                        try {
                            contentResolver.openInputStream(uri)?.use { input ->
                                modelFile.outputStream().use { output ->
                                    input.copyTo(output)
                                }
                            }
                            true
                        } catch (e: Exception) {
                            status = "Copy failed: ${e.message}"
                            false
                        }
                    }

                    if (!copied) return@launch

                    status = "Loading model..."

                    val loaded = withContext(Dispatchers.Default) {
                        engine.nativeLoadModel(modelFile.absolutePath, 2048, 4)
                    }

                    if (!loaded) {
                        status = "Model failed to load"
                        return@launch
                    }

                    status = "Model loaded. Generating...\n\n"

                    val prompt =
                        "<|im_start|>system\n" +
                        "You are Kairo, a helpful AI assistant.<|im_end|>\n" +
                        "<|im_start|>user\n" +
                        "Hello, who are you?<|im_end|>\n" +
                        "<|im_start|>assistant\n"

                    withContext(Dispatchers.Default) {
                        var piece = engine.nativeGenerateToken(prompt, true)
                        var count = 0

                        while (piece.isNotEmpty() && count < 200) {
                            val text = piece
                            withContext(Dispatchers.Main) {
                                status += text
                            }
                            count++
                            piece = engine.nativeGenerateToken(null, false)
                        }
                    }
                }
            }

            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    Column(modifier = Modifier.fillMaxSize().padding(24.dp)) {
                        Button(
                            onClick = { pickFile.launch(arrayOf("*/*")) },
                            enabled = engine != null
                        ) {
                            Text("Choose GGUF model file")
                        }
                        Text(status, modifier = Modifier.padding(top = 24.dp))
                    }
                }
            }
        }
    }
}
