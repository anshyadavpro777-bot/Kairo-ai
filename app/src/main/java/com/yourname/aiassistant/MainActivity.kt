package com.yourname.aiassistant

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
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
        enableEdgeToEdge()

        val engine: LlamaEngine? = try {
            LlamaEngine()
        } catch (e: UnsatisfiedLinkError) {
            null
        }

        setContent {
            var status by remember {
                mutableStateOf(
                    if (engine == null) "Native library failed to load"
                    else "Choose a model file, or load the saved one"
                )
            }
            var question by remember { mutableStateOf("") }
            var reply by remember { mutableStateOf("") }
            var input by remember { mutableStateOf("") }
            var modelReady by remember { mutableStateOf(false) }
            var generating by remember { mutableStateOf(false) }

            val scope = rememberCoroutineScope()
            val scrollState = rememberScrollState()
            val savedModel = File(filesDir, "model.gguf")

            LaunchedEffect(reply) {
                scrollState.scrollTo(scrollState.maxValue)
            }

            suspend fun loadModel(file: File) {
                modelReady = false
                status = "Loading model..."
                val loaded = withContext(Dispatchers.Default) {
                    engine?.nativeLoadModel(file.absolutePath, 2048, 4) == true
                }
                modelReady = loaded
                status = if (loaded) "Model ready. Type a message below."
                else "Model failed to load"
            }

            fun send() {
                val text = input.trim()
                if (engine == null || text.isEmpty() || !modelReady || generating) return

                input = ""
                question = text
                reply = ""
                generating = true

                scope.launch {
                    val prompt =
                        "<|im_start|>system\n" +
                        "You are Kairo, a helpful AI assistant.<|im_end|>\n" +
                        "<|im_start|>user\n" +
                        text + "<|im_end|>\n" +
                        "<|im_start|>assistant\n"

                    val finalText = withContext(Dispatchers.Default) {
                        var raw = ""
                        var piece = engine.nativeGenerateToken(prompt, true)
                        var count = 0

                        while (piece.isNotEmpty() && count < 400) {
                            raw += piece
                            if (raw.contains("<|im_end|>")) break
                            val shown = raw
                            withContext(Dispatchers.Main) {
                                reply = shown
                            }
                            count++
                            piece = engine.nativeGenerateToken(null, false)
                        }
                        raw.replace("<|im_end|>", "").trim()
                    }

                    reply = if (finalText.isEmpty()) "(No reply - try again)" else finalText
                    generating = false
                }
            }

            val pickFile = rememberLauncherForActivityResult(
                contract = ActivityResultContracts.OpenDocument()
            ) { uri: Uri? ->
                if (engine == null) return@rememberLauncherForActivityResult
                if (uri == null) {
                    status = "No file selected"
                    return@rememberLauncherForActivityResult
                }

                scope.launch {
                    modelReady = false
                    status = "Copying model into the app... this can take a minute"

                    withContext(Dispatchers.Default) {
                        engine.nativeUnload()
                    }

                    val error: String? = withContext(Dispatchers.IO) {
                        try {
                            contentResolver.openInputStream(uri)?.use { src ->
                                savedModel.outputStream().use { dst ->
                                    src.copyTo(dst)
                                }
                            }
                            null
                        } catch (e: Exception) {
                            e.message ?: "unknown error"
                        }
                    }

                    if (error != null) {
                        status = "Copy failed: $error"
                        return@launch
                    }

                    loadModel(savedModel)
                }
            }

            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .systemBarsPadding()
                            .imePadding()
                            .padding(16.dp)
                    ) {
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(
                                onClick = { pickFile.launch(arrayOf("*/*")) },
                                enabled = engine != null && !generating
                            ) {
                                Text("Choose model")
                            }
                            Button(
                                onClick = { scope.launch { loadModel(savedModel) } },
                                enabled = engine != null && !generating && savedModel.exists()
                            ) {
                                Text("Load saved")
                            }
                        }

                        Text(status, modifier = Modifier.padding(top = 8.dp))

                        Column(
                            modifier = Modifier
                                .weight(1f)
                                .fillMaxWidth()
                                .verticalScroll(scrollState)
                                .padding(top = 12.dp)
                        ) {
                            if (question.isNotEmpty()) {
                                Text("You: $question")
                                Text(
                                    "Kairo: $reply",
                                    modifier = Modifier.padding(top = 12.dp)
                                )
                            }
                        }

                        Row(verticalAlignment = Alignment.CenterVertically) {
                            OutlinedTextField(
                                value = input,
                                onValueChange = { input = it },
                                modifier = Modifier.weight(1f),
                                placeholder = { Text("Type a message...") },
                                enabled = modelReady && !generating
                            )
                            Button(
                                onClick = { send() },
                                enabled = modelReady && !generating && input.isNotBlank(),
                                modifier = Modifier.padding(start = 8.dp)
                            ) {
                                Text("Send")
                            }
                        }
                    }
                }
            }
        }
    }
}