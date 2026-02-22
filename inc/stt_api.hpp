/**
 * EvoAudioSDK - Evo ASR Engine SDK
 *
 * 语音识别引擎适配层，提供统一的 C++ 接口。
 *
 * 使用示例 1 - 文件识别（阻塞模式）:
 *
 *   auto engine = std::make_shared<Evo::AsrEngine>();
 *   auto result = engine->Call("test.wav");
 *   if (result) {
 *       std::cout << result->GetSentence().text << std::endl;
 *   }
 *
 * 使用示例 2 - 带配置的初始化:
 *
 *   Evo::AsrConfig config;
 *   config.language = "en";
 *   config.punctuation = false;
 *   auto engine = std::make_shared<Evo::AsrEngine>(config);
 *
 * 使用示例 3 - 流式识别:
 *
 *   auto engine = std::make_shared<Evo::AsrEngine>();
 *   engine->SetCallback(std::make_shared<MyCallback>());
 *   engine->Start();
 *   while (recording) {
 *       engine->SendAudioFrame(audio_data);
 *   }
 *   engine->Stop();
 */

#ifndef STT_API_HPP
#define STT_API_HPP

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <cstdint>

// Forward declaration of internal types
namespace asr {
    class ASREngine;
    struct RecognitionResult;
    struct ErrorInfo;
}

namespace Evo {

// =============================================================================
// AsrConfig - ASR 配置
// =============================================================================

/**
 * @brief ASR 引擎配置
 *
 *
 */
struct AsrConfig {
    std::string engine = "sensevoice";  ///< 引擎类型 ("sensevoice", "funasr", "whisper")
    std::string model_dir;              ///< 模型目录路径，空则使用默认路径
    std::string language = "zh";        ///< 语言 ("zh", "en", "ja", "ko", "yue", "auto")
    bool punctuation = true;            ///< 自动添加标点
    int sample_rate = 16000;            ///< 音频采样率 (Hz)

    /// @brief 创建默认配置
    static AsrConfig Default() {
        return AsrConfig();
    }

    /// @brief 创建 SenseVoice 配置
    /// @param model_dir 模型目录路径
    static AsrConfig SenseVoice(const std::string& model_dir = "~/.cache/sensevoice") {
        AsrConfig config;
        config.engine = "sensevoice";
        config.model_dir = model_dir;
        return config;
    }
};

// =============================================================================
// Sentence - 句子结果
// =============================================================================

struct Sentence {
    std::string text;        // 识别文本
    int begin_time = 0;      // 开始时间 (毫秒)
    int end_time = 0;        // 结束时间 (毫秒)
    float confidence = 0.0f;  // 置信度 [0.0, 1.0]
};

// =============================================================================
// RecognitionResult - 识别结果
// =============================================================================

class RecognitionResult {
public:
    RecognitionResult();
    ~RecognitionResult();

    // 禁止拷贝，允许移动
    RecognitionResult(const RecognitionResult&) = delete;
    RecognitionResult& operator=(const RecognitionResult&) = delete;
    RecognitionResult(RecognitionResult&&) noexcept;
    RecognitionResult& operator=(RecognitionResult&&) noexcept;

    /// @brief 获取主句子结果
    /// @return 句子结构体
    Sentence GetSentence() const;

    /// @brief 获取所有句子结果
    /// @return 句子列表
    std::vector<Sentence> GetSentences() const;

    /// @brief 是否为句子结束（最终结果）
    /// @return true 表示最终结果，false 表示中间结果
    bool IsSentenceEnd() const;

    /// @brief 获取请求 ID
    /// @return 请求标识符
    std::string GetRequestId() const;

    /// @brief 获取完整识别文本
    /// @return 所有句子拼接的文本
    std::string GetText() const;

    /// @brief 是否为空结果
    /// @return true 表示无识别内容
    bool IsEmpty() const;

    /// @brief 获取音频时长
    /// @return 音频时长（毫秒）
    int GetAudioDuration() const;

    /// @brief 获取处理时间
    /// @return 处理耗时（毫秒）
    int GetProcessingTime() const;

    /// @brief 获取实时率 (RTF)
    /// @return 处理时间 / 音频时长
    float GetRTF() const;

private:
    friend class AsrEngine;
    friend class CallbackAdapter;

    // 内部设置方法
    void setFromInternal(const asr::RecognitionResult& internal);
    void setFinal(bool is_final);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// =============================================================================
// AsrEngineCallback - 回调接口
// =============================================================================

/**
 * @brief ASR 引擎回调接口（多态基类）
 *
 * 用户可继承此类并重写虚函数，以接收流式识别过程中的事件通知。
 * 通过 AsrEngine::SetCallback() 注册回调实例。
 *
 * ## 回调调用链（调用顺序）
 *
 * 流式识别的回调调用顺序如下：
 *
 * ```
 *   Start()
 *      │
 *      ▼
 *   OnOpen()              ← 识别会话开始
 *      │
 *      ▼
 *   SendAudioFrame() ───► OnEvent()  ← 每次有识别结果时触发（可能多次）
 *      │                     │
 *      │                     ├─ is_final=false: 中间结果（实时转写）
 *      │                     └─ is_final=true:  句子结束（最终结果）
 *      ▼
 *   Stop()
 *      │
 *      ▼
 *   OnComplete()          ← 识别正常完成
 *      │
 *      ▼
 *   OnClose()             ← 会话关闭
 * ```
 *
 * ## 错误处理
 *
 * 如果识别过程中发生错误，调用链为：
 * ```
 *   OnOpen() → ... → OnError() → OnClose()
 * ```
 *
 * ## 线程安全
 *
 * - 回调可能在引擎内部线程中被调用，实现时需注意线程安全
 * - 不要在回调中执行耗时操作，以免阻塞识别流程
 *
 * ## 使用示例
 *
 * ```cpp
 * class MyCallback : public AsrEngineCallback {
 * public:
 *     void OnOpen() override {
 *         std::cout << "开始识别" << std::endl;
 *     }
 *
 *     void OnEvent(std::shared_ptr<RecognitionResult> result) override {
 *         if (result->IsSentenceEnd()) {
 *             std::cout << "最终: " << result->GetText() << std::endl;
 *         } else {
 *             std::cout << "中间: " << result->GetText() << std::endl;
 *         }
 *     }
 *
 *     void OnComplete() override {
 *         std::cout << "识别完成" << std::endl;
 *     }
 *
 *     void OnError(std::shared_ptr<RecognitionResult> result) override {
 *         std::cerr << "错误: " << result->GetText() << std::endl;
 *     }
 *
 *     void OnClose() override {
 *         std::cout << "会话关闭" << std::endl;
 *     }
 * };
 *
 * // 注册回调
 * auto engine = std::make_shared<AsrEngine>();
 * engine->SetCallback(std::make_shared<MyCallback>());
 * ```
 */
class AsrEngineCallback {
public:
    virtual ~AsrEngineCallback() = default;

    /// @brief 连接建立成功，识别会话开始
    /// @note 在 Start() 调用后触发
    virtual void OnOpen() {}

    /// @brief 收到识别结果
    /// @param result 识别结果对象
    /// @note 可能被多次调用：
    ///       - IsSentenceEnd()=false: 中间结果（实时转写，文本可能变化）
    ///       - IsSentenceEnd()=true:  最终结果（句子结束，文本稳定）
    virtual void OnEvent(std::shared_ptr<RecognitionResult> result) {}

    /// @brief 识别任务正常完成
    /// @note 在 Stop() 调用后、OnClose() 之前触发
    virtual void OnComplete() {}

    /// @brief 发生错误
    /// @param result 包含错误信息的结果对象，可通过 GetText() 获取错误描述
    /// @note 错误发生后会自动调用 OnClose()
    virtual void OnError(std::shared_ptr<RecognitionResult> result) {}

    /// @brief 会话关闭
    /// @note 无论正常结束还是错误，最后都会调用此方法
    virtual void OnClose() {}
};

// =============================================================================
// AsrEngine - ASR 引擎
// =============================================================================

class AsrEngine {
public:
    /// @brief 构造 ASR 引擎（使用默认配置）
    /// @param engine 引擎类型 ("sensevoice", "funasr", "whisper")
    /// @param model_dir 模型目录路径，空则使用默认路径
    explicit AsrEngine(const std::string& engine = "sensevoice",
        const std::string& model_dir = "");

    /// @brief 构造 ASR 引擎（使用配置结构体）
    /// @param config 配置对象
    explicit AsrEngine(const AsrConfig& config);

    /// @brief 析构函数
    virtual ~AsrEngine();

    // 禁止拷贝
    AsrEngine(const AsrEngine&) = delete;
    AsrEngine& operator=(const AsrEngine&) = delete;

    // -------------------------------------------------------------------------
    // 非流式调用（阻塞）
    // -------------------------------------------------------------------------

    /// @brief 识别音频文件（阻塞直到完成）
    /// @param file_path 音频文件路径
    /// @param phrase_id 热词组 ID（可选，暂未实现）
    /// @return 识别结果，失败返回 nullptr
    std::shared_ptr<RecognitionResult> Call(
        const std::string& file_path,
        const std::string& phrase_id = "");

    /// @brief 识别音频数据（16-bit PCM）
    /// @param audio 音频数据
    /// @param sample_rate 采样率（默认 16000）
    /// @return 识别结果，失败返回 nullptr
    std::shared_ptr<RecognitionResult> Recognize(
        const std::vector<int16_t>& audio,
        int sample_rate = 16000);

    /// @brief 识别音频数据（float PCM, [-1.0, 1.0]）
    /// @param audio 音频数据
    /// @param sample_rate 采样率（默认 16000）
    /// @return 识别结果，失败返回 nullptr
    std::shared_ptr<RecognitionResult> Recognize(
        const std::vector<float>& audio,
        int sample_rate = 16000);

    // -------------------------------------------------------------------------
    // 流式调用
    // -------------------------------------------------------------------------

    /// @brief 设置回调监听器
    /// @param callback 回调对象
    void SetCallback(std::shared_ptr<AsrEngineCallback> callback);

    /// @brief 开始流式识别
    /// @param phrase_id 热词组 ID（可选，暂未实现）
    void Start(const std::string& phrase_id = "");

    /// @brief 发送音频帧
    /// @param data 音频数据（16kHz, 16bit, mono, PCM）
    void SendAudioFrame(const std::vector<uint8_t>& data);

    /// @brief 刷新缓冲区并立即识别（不关闭会话）
    /// @note 用于用户 VAD 检测到句子结束时手动触发识别。
    ///       识别结果通过回调返回，会话保持活跃可继续发送音频。
    void Flush();

    /// @brief 停止流式识别
    void Stop();

    // -------------------------------------------------------------------------
    // 动态配置
    // -------------------------------------------------------------------------

    /// @brief 设置识别语言
    /// @param language 语言代码 ("zh", "en", "ja", "ko", "yue", "auto")
    void SetLanguage(const std::string& language);

    /// @brief 设置是否自动添加标点
    /// @param enabled true 启用，false 禁用
    void SetPunctuation(bool enabled);

    /// @brief 获取当前配置
    /// @return 配置对象
    AsrConfig GetConfig() const;

    // -------------------------------------------------------------------------
    // 辅助方法
    // -------------------------------------------------------------------------

    /// @brief 获取最后一次请求 ID
    /// @return 请求 ID
    std::string GetLastRequestId();

    /// @brief 获取首包延迟
    /// @return 延迟时间（毫秒）
    int GetFirstPackageDelay();

    /// @brief 获取尾包延迟
    /// @return 延迟时间（毫秒）
    int GetLastPackageDelay();

    /// @brief 获取最后一次识别结果的 JSON 响应
    /// @return JSON 格式字符串
    std::string GetResponse();

    /// @brief 检查引擎是否已初始化
    /// @return true 表示已初始化
    bool IsInitialized() const;

    /// @brief 获取引擎类型名称
    /// @return 引擎名称
    std::string GetEngineName() const;

private:
    friend class CallbackAdapter;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace Evo

#endif  // STT_API_HPP
