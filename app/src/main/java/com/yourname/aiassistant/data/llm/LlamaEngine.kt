package com.yourname.aiassistant.data.llm

class LlamaEngine {

    companion object {
        init {
            System.loadLibrary("llama-bridge")
        }
    }

    external fun nativeLoadModel(modelPath: String, contextSize: Int, threads: Int): Boolean

    external fun nativeGenerateToken(prompt: String?, isFirst: Boolean): String

    external fun nativeStopGeneration()

    external fun nativeResetConversation()

    external fun nativeUnload()
}
