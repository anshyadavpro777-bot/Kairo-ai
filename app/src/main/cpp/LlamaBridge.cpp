#include <jni.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>

#include <android/log.h>

#include "llama.h"

#define LOG_TAG "KAIRO_LLM"

#define LOGI(...) \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#define LOGE(...) \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static llama_model* g_model = nullptr;
static llama_context* g_context = nullptr;
static llama_sampler* g_sampler = nullptr;

static const llama_vocab* g_vocab = nullptr;

static int32_t g_n_past = 0;

static llama_token g_last_token = -1;

static std::mutex g_mutex;

static std::atomic<bool> g_stop_requested(false);

static void reset_state() {

    g_n_past = 0;

    g_last_token = -1;

    g_stop_requested = false;
}

static bool tokenize_text(
        const std::string& text,
        std::vector<llama_token>& tokens) {

    if (g_vocab == nullptr) {

        LOGE("Vocabulary is not available");

        return false;
    }

    int32_t token_count = -llama_tokenize(
            g_vocab,
            text.c_str(),
            static_cast<int32_t>(text.size()),
            nullptr,
            0,
            true,
            true
    );

    if (token_count <= 0) {

        LOGE("Could not calculate token count");

        return false;
    }

    tokens.resize(token_count);

    int32_t result = llama_tokenize(
            g_vocab,
            text.c_str(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            token_count,
            true,
            true
    );

    if (result < 0) {

        LOGE("Tokenization failed");

        tokens.clear();

        return false;
    }

    tokens.resize(result);

    return true;
}

static bool decode_tokens(
        const std::vector<llama_token>& tokens) {

    if (tokens.empty()) {

        return true;
    }

    if (g_context == nullptr) {

        LOGE("Context is not available");

        return false;
    }

    llama_batch batch = llama_batch_init(
            static_cast<int32_t>(tokens.size()),
            0,
            1
    );

    if (batch.token == nullptr) {

        LOGE("Could not create llama batch");

        return false;
    }

    for (size_t i = 0; i < tokens.size(); ++i) {

        batch.token[i] = tokens[i];

        batch.pos[i] =
                g_n_past +
                static_cast<int32_t>(i);

        batch.n_seq_id[i] = 1;

        batch.seq_id[i][0] = 0;

        batch.logits[i] =
                (i == tokens.size() - 1);
    }

    batch.n_tokens =
            static_cast<int32_t>(tokens.size());

    int result =
            llama_decode(
                    g_context,
                    batch
            );

    llama_batch_free(batch);

    if (result != 0) {

        LOGE(
                "llama_decode failed: %d",
                result
        );

        return false;
    }

    g_n_past +=
            static_cast<int32_t>(tokens.size());

    return true;
}

static std::string token_to_text(
        llama_token token) {

    if (g_vocab == nullptr) {

        return "";
    }

    char buffer[512];

    int32_t size =
            llama_token_to_piece(
                    g_vocab,
                    token,
                    buffer,
                    sizeof(buffer),
                    0,
                    true
            );

    if (size < 0) {

        LOGE("Token conversion failed");

        return "";
    }

    return std::string(
            buffer,
            size
    );
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_yourname_aiassistant_data_llm_LlamaEngine_nativeLoadModel(
        JNIEnv* env,
        jobject,
        jstring modelPath,
        jint contextSize,
        jint threads) {

    std::lock_guard<std::mutex> lock(g_mutex);

    if (modelPath == nullptr) {

        LOGE("Model path is null");

        return JNI_FALSE;
    }

    if (contextSize <= 0) {

        LOGE("Invalid context size");

        return JNI_FALSE;
    }

    if (threads <= 0) {

        LOGE("Invalid thread count");

        return JNI_FALSE;
    }

    if (g_sampler != nullptr) {

        llama_sampler_free(g_sampler);

        g_sampler = nullptr;
    }

    if (g_context != nullptr) {

        llama_free(g_context);

        g_context = nullptr;
    }

    if (g_model != nullptr) {

        llama_model_free(g_model);

        g_model = nullptr;
    }

    g_vocab = nullptr;

    reset_state();

    llama_backend_init();

    const char* path =
            env->GetStringUTFChars(
                    modelPath,
                    nullptr
            );

    if (path == nullptr) {

        LOGE("Could not read model path");

        llama_backend_free();

        return JNI_FALSE;
    }

    std::string model_file(path);

    env->ReleaseStringUTFChars(
            modelPath,
            path
    );

    LOGI(
            "Kairo loading model: %s",
            model_file.c_str()
    );

    llama_model_params model_params =
            llama_model_default_params();

    model_params.n_gpu_layers = 0;

    model_params.use_mmap = true;

    g_model =
            llama_model_load_from_file(
                    model_file.c_str(),
                    model_params
            );

    if (g_model == nullptr) {

        LOGE("Failed to load GGUF model");

        llama_backend_free();

        return JNI_FALSE;
    }

    g_vocab =
            llama_model_get_vocab(
                    g_model
            );

    if (g_vocab == nullptr) {

        LOGE("Could not get model vocabulary");

        llama_model_free(g_model);

        g_model = nullptr;

        llama_backend_free();

        return JNI_FALSE;
    }

    llama_context_params context_params =
            llama_context_default_params();

    context_params.n_ctx =
            static_cast<uint32_t>(
                    contextSize
            );

    context_params.n_batch = 128;

    context_params.n_ubatch = 128;

    context_params.n_threads =
            threads;

    context_params.n_threads_batch =
            threads;

    g_context =
            llama_init_from_model(
                    g_model,
                    context_params
            );

    if (g_context == nullptr) {

        LOGE("Failed to create llama context");

        llama_model_free(g_model);

        g_model = nullptr;

        g_vocab = nullptr;

        llama_backend_free();

        return JNI_FALSE;
    }

    llama_sampler_chain_params sampler_params =
            llama_sampler_chain_default_params();

    g_sampler =
            llama_sampler_chain_init(
                    sampler_params
            );

    if (g_sampler == nullptr) {

        LOGE("Failed to create sampler");

        llama_free(g_context);

        g_context = nullptr;

        llama_model_free(g_model);

        g_model = nullptr;

        g_vocab = nullptr;

        llama_backend_free();

        return JNI_FALSE;
    }

    llama_sampler_chain_add(
            g_sampler,
            llama_sampler_init_temp(
                    0.7f
            )
    );

    llama_sampler_chain_add(
            g_sampler,
            llama_sampler_init_top_k(
                    40
            )
    );

    llama_sampler_chain_add(
            g_sampler,
            llama_sampler_init_top_p(
                    0.95f,
                    1
            )
    );

    llama_sampler_chain_add(
            g_sampler,
            llama_sampler_init_dist(
                    LLAMA_DEFAULT_SEED
            )
    );

    reset_state();

    LOGI("Kairo model loaded successfully");

    return JNI_TRUE;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_yourname_aiassistant_data_llm_LlamaEngine_nativeGenerateToken(
        JNIEnv* env,
        jobject,
        jstring prompt,
        jboolean isFirst) {

    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_model == nullptr ||
        g_context == nullptr ||
        g_sampler == nullptr ||
        g_vocab == nullptr) {

        LOGE("Kairo engine is not loaded");

        return env->NewStringUTF("");
    }

    if (g_stop_requested) {

        g_last_token = -1;

        return env->NewStringUTF("");
    }

    if (isFirst) {

        if (prompt == nullptr) {

            LOGE("Prompt is null");

            return env->NewStringUTF("");
        }

        const char* prompt_chars =
                env->GetStringUTFChars(
                        prompt,
                        nullptr
                );

        if (prompt_chars == nullptr) {

            return env->NewStringUTF("");
        }

        std::string prompt_text(
                prompt_chars
        );

        env->ReleaseStringUTFChars(
                prompt,
                prompt_chars
        );

        LOGI(
                "Starting generation. Prompt length: %zu",
                prompt_text.size()
        );

        llama_memory_clear(
                llama_get_memory(g_context),
                true
        );

        reset_state();

        std::vector<llama_token> prompt_tokens;

        if (!tokenize_text(
                prompt_text,
                prompt_tokens)) {

            return env->NewStringUTF("");
        }

        int32_t context_size =
                static_cast<int32_t>(
                        llama_n_ctx(g_context)
                );

        if (prompt_tokens.size() >=
            static_cast<size_t>(context_size)) {

            LOGE(
                    "Prompt too large: %zu / %d tokens",
                    prompt_tokens.size(),
                    context_size
            );

            return env->NewStringUTF("");
        }

        if (!decode_tokens(
                prompt_tokens)) {

            return env->NewStringUTF("");
        }
    }

    else {

        if (g_last_token < 0) {

            return env->NewStringUTF("");
        }

        if (g_stop_requested) {

            g_last_token = -1;

            return env->NewStringUTF("");
        }

        int32_t context_size =
                static_cast<int32_t>(
                        llama_n_ctx(g_context)
                );

        if (g_n_past >= context_size) {

            LOGI(
                    "Kairo context limit reached"
            );

            g_last_token = -1;

            return env->NewStringUTF("");
        }

        std::vector<llama_token> token = {
                g_last_token
        };

        if (!decode_tokens(token)) {

            g_last_token = -1;

            return env->NewStringUTF("");
        }
    }

    llama_token next_token =
            llama_sampler_sample(
                    g_sampler,
                    g_context,
                    -1
            );

    llama_sampler_accept(
            g_sampler,
            next_token
    );

    if (llama_vocab_is_eog(
            g_vocab,
            next_token)) {

        LOGI(
                "Kairo generation finished"
        );

        g_last_token = -1;

        return env->NewStringUTF("");
    }

    g_last_token = next_token;

    std::string text =
            token_to_text(
                    next_token
            );

    if (text.empty()) {

        return env->NewStringUTF("");
    }

    return env->NewStringUTF(
            text.c_str()
    );
}

extern "C"
JNIEXPORT void JNICALL
Java_com_yourname_aiassistant_data_llm_LlamaEngine_nativeStopGeneration(
        JNIEnv*,
        jobject) {

    g_stop_requested = true;

    LOGI(
            "Kairo generation stop requested"
    );
}

extern "C"
JNIEXPORT void JNICALL
Java_com_yourname_aiassistant_data_llm_LlamaEngine_nativeResetConversation(
        JNIEnv*,
        jobject) {

    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_context == nullptr) {

        return;
    }

    llama_memory_clear(
            llama_get_memory(g_context),
            true
    );

    reset_state();

    LOGI(
            "Kairo conversation reset"
    );
}

extern "C"
JNIEXPORT void JNICALL
Java_com_yourname_aiassistant_data_llm_LlamaEngine_nativeUnload(
        JNIEnv*,
        jobject) {

    std::lock_guard<std::mutex> lock(g_mutex);

    g_stop_requested = true;

    if (g_sampler != nullptr) {

        llama_sampler_free(
                g_sampler
        );

        g_sampler = nullptr;
    }

    if (g_context != nullptr) {

        llama_free(
                g_context
        );

        g_context = nullptr;
    }

    if (g_model != nullptr) {

        llama_model_free(
                g_model
        );

        g_model = nullptr;
    }

    g_vocab = nullptr;

    reset_state();

    llama_backend_free();

    LOGI(
            "Kairo engine unloaded"
    );
}
