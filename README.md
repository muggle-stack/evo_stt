# STT - 离线语音识别引擎

基于 SenseVoice 的离线语音识别框架，支持多语言、流式/非流式识别，提供 C++ 和 Python 接口。

## 特性

- SenseVoice ONNX 离线推理，无需网络连接
- 支持中文、英文、日语、韩语、粤语及自动语言检测
- 流式识别（回调模式）与非流式识别（阻塞模式）
- Flush 机制：流式模式下可随时触发识别，无需关闭会话
- 自动标点符号恢复
- Python 绑定（pybind11），支持 NumPy 数组输入

## 依赖

### 系统依赖

| 平台 | 安装命令 |
|------|---------|
| Ubuntu/Debian | `sudo apt install libsndfile1-dev libfftw3-dev libcurl4-openssl-dev` |
| macOS | `brew install libsndfile fftw curl onnxruntime` |

ONNX Runtime 需手动安装或通过包管理器安装。可选：`portaudio`（流式示例）、`pybind11`（Python 绑定）。

### 模型文件

首次运行自动下载到 `~/.cache/sensevoice/`：

| 文件 | 说明 |
|------|------|
| `model_quant_optimized.onnx` | 量化 SenseVoice 模型 |
| `vocab.txt` | 词表 |
| `config.json` | 模型配置 |
| `am.mvn` | CMVN 归一化参数 |

## 编译

```bash
mkdir -p build && cd build
cmake .. && make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

CMake 选项：

| 选项 | 默认 | 说明 |
|------|------|------|
| `BUILD_STREAM_DEMO` | OFF | 构建流式识别示例（需 audio 模块 + PortAudio） |
| `BUILD_PYTHON_BINDINGS` | ON | 构建 Python 绑定 |

编译产物：

```
build/lib/libstt.a          # 框架库
build/lib/libsensevoice.a   # SenseVoice 后端库
build/bin/stt_test           # 文件识别示例
build/bin/stream_demo        # 流式识别示例（可选）
```

## 快速开始

### C++

```cpp
#include "stt_api.hpp"

int main() {
    auto config = Evo::AsrConfig::SenseVoice();
    config.language = "zh";
    auto engine = std::make_shared<Evo::AsrEngine>(config);
    auto result = engine->Call("audio.wav");
    std::cout << result->GetText() << std::endl;
    return 0;
}
```

### Python

```bash
cd modules/stt/python && pip install -e .
```

```python
import evo_asr

text = evo_asr.recognize_file("audio.wav")
print(text)
```

## 示例

```bash
# C++ 文件识别
./build/bin/stt_test audio.wav
./build/bin/stt_test audio.wav ~/.cache/sensevoice

# C++ 流式识别（需 BUILD_STREAM_DEMO=ON）
./build/bin/stream_demo -l          # 列出音频设备
./build/bin/stream_demo 0 30        # 设备 0，录音 30 秒
./build/bin/stream_demo -i 5        # 每 5 秒 Flush 一次

# Python
python examples/static_file_demo.py audio.wav
python examples/stream_demo.py
```

## API 概览

### 核心类（命名空间 `Evo::`）

| 类 | 说明 |
|----|------|
| `AsrEngine` | 语音识别引擎，支持文件/内存/流式识别 |
| `AsrConfig` | 引擎配置，提供 `SenseVoice()` 工厂方法 |
| `RecognitionResult` | 识别结果，含文本、时间戳、置信度、RTF |
| `AsrEngineCallback` | 流式识别回调接口（OnOpen/OnEvent/OnComplete/OnError/OnClose） |

### 关键方法

| 方法 | 说明 |
|------|------|
| `Call(file_path)` | 文件识别（阻塞） |
| `Recognize(audio, sample_rate)` | 内存音频识别（阻塞） |
| `Start()` / `SendAudioFrame()` / `Stop()` | 流式识别 |
| `Flush()` | 流式模式下立即触发识别，不关闭会话 |

详细文档见 [API.md](API.md)

## 配置参数

### AsrConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `engine` | `string` | `"sensevoice"` | 引擎类型 |
| `model_dir` | `string` | `"~/.cache/sensevoice"` | 模型目录 |
| `language` | `string` | `"zh"` | 语言：`zh`/`en`/`ja`/`ko`/`yue`/`auto` |
| `punctuation` | `bool` | `true` | 启用自动标点 |
| `sample_rate` | `int` | `16000` | 输入采样率 (Hz) |

## CMake 集成

```cmake
add_subdirectory(modules/stt)
target_link_libraries(your_target PRIVATE stt)
target_include_directories(your_target PRIVATE ${STT_SOURCE_DIR}/inc)
```

## 许可证

MIT License - 详见 [LICENSE](LICENSE)
