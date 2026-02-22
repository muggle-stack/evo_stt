#ifndef ASR_HPP
#define ASR_HPP

// =============================================================================
// ASR Framework - 主包含文件
// =============================================================================
//
// 只需包含此文件即可使用完整的ASR框架
//
//   #include <asr/asr.hpp>
//
// 快速开始:
//
//   // 1. 创建引擎
//   asr::ASREngine engine;
//
//   // 2. 配置 (使用SenseVoice本地模型)
//   auto config = asr::ASRConfig::sensevoice("~/.cache/sensevoice");
//
//   // 3. 初始化
//   auto err = engine.initialize(config);
//   if (!err.isOk()) {
//       std::cerr << "Init failed: " << err.message << std::endl;
//       return -1;
//   }
//
//   // 4. 识别
//   auto result = engine.recognize(audio_data, sample_count);
//   std::cout << "Result: " << result.getText() << std::endl;
//

#include <string>

// 核心类型
#include "asr_types.hpp"

// 配置
#include "asr_config.hpp"

// 回调接口
#include "asr_callback.hpp"

// 主引擎
#include "asr_engine.hpp"

// 后端接口 (通常不需要直接使用)
#include "backends/asr_backend.hpp"

namespace asr {

// =============================================================================
// 版本信息
// =============================================================================

constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

inline std::string getVersionString() {
    return std::to_string(VERSION_MAJOR) + "." +
        std::to_string(VERSION_MINOR) + "." +
        std::to_string(VERSION_PATCH);
}

// =============================================================================
// 使用示例
// =============================================================================
//
// 示例 1: 简单离线识别
// ---------------------
//
//   #include <asr/asr.hpp>
//
//   int main() {
//       asr::ASREngine engine;
//
//       // SenseVoice本地模型
//       auto config = asr::ASRConfig::sensevoice("~/.cache/sensevoice");
//       engine.initialize(config);
//
//       // 读取音频文件
//       auto result = engine.recognizeFile("test.wav");
//       std::cout << result.getText() << std::endl;
//
//       return 0;
//   }
//
// 示例 2: 流式识别 + 回调
// ------------------------
//
//   #include <asr/asr.hpp>
//
//   int main() {
//       asr::ASREngine engine;
//
//       // 配置流式模式
//       auto config = asr::ASRConfig::sensevoice("~/.cache/sensevoice")
//                        .withStreaming(100);  // 100ms 块大小
//
//       // 设置回调
//       engine.setCallback(asr::LambdaCallback::create()
//           .onResult([](const asr::RecognitionResult& r) {
//               for (const auto& s : r.sentences) {
//                   std::cout << (s.is_final ? "[FINAL] " : "[PARTIAL] ")
//                             << s.text << std::endl;
//               }
//           })
//           .onError([](const asr::ErrorInfo& e) {
//               std::cerr << "Error: " << e.message << std::endl;
//           })
//           .build());
//
//       engine.initialize(config);
//
//       // 开始流式识别
//       engine.start();
//
//       // 模拟音频输入
//       while (has_more_audio()) {
//           auto chunk = read_audio_chunk();
//           engine.sendAudio(chunk.data(), chunk.size());
//       }
//
//       // 结束
//       engine.stop();
//
//       return 0;
//   }
//
// 示例 3: 切换不同后端
// ---------------------
//
//   #include <asr/asr.hpp>
//
//   int main() {
//       asr::ASREngine engine;
//
//       // 检查可用后端
//       for (auto type : asr::ASREngine::getAvailableBackends()) {
//           std::cout << "Available: " << asr::backendTypeToString(type) << std::endl;
//       }
//
//       // 使用FunASR云端 (未来扩展)
//       auto config = asr::ASRConfig::funasrCloud("your-api-key");
//       config.language = asr::Language::ZH;
//       config.vad_enabled = true;
//
//       engine.initialize(config);
//       // ...
//
//       return 0;
//   }
//
// 示例 4: 词级别时间戳
// ---------------------
//
//   auto config = asr::ASRConfig::sensevoice("~/.cache/sensevoice")
//                    .withWordTimestamps();
//
//   engine.initialize(config);
//   auto result = engine.recognizeFile("test.wav");
//
//   for (const auto& sentence : result.sentences) {
//       std::cout << "Sentence: " << sentence.text << std::endl;
//       std::cout << "  Time: " << sentence.begin_time_ms << "ms - "
//                 << sentence.end_time_ms << "ms" << std::endl;
//
//       for (const auto& word : sentence.words) {
//           std::cout << "  Word: " << word.text
//                     << " [" << word.begin_time_ms << "-" << word.end_time_ms << "ms]"
//                     << " conf=" << word.confidence << std::endl;
//       }
//   }
//

}  // namespace asr

#endif  // ASR_HPP
