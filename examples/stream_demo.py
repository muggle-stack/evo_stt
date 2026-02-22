#!/usr/bin/env python3
"""
ASR 流式识别示例

依赖:
    1. asr 模块 (stt/build/python)
    2. evo_audio 模块 (audio/python)
    3. numpy
    4. scipy (可选，用于高质量重采样)

编译 audio 组件:
    cd audio && mkdir -p build && cd build
    cmake .. && make -j$(nproc)

Usage:
    python stream_demo.py                 # 默认设备，持续识别
    python stream_demo.py -l              # 列出音频设备
    python stream_demo.py -d 0            # 使用设备 0
    python stream_demo.py --duration 5    # 每次录制 5 秒
"""

import sys
import time
import argparse

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from scipy import signal as scipy_signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def import_modules():
    """导入必要模块"""
    # 检查 numpy
    if not HAS_NUMPY:
        print("错误: numpy 未安装，请运行: pip install numpy")
        sys.exit(1)

    # 导入 asr_py（C++ 绑定模块）
    try:
        import asr_py as asr
    except ImportError as e:
        print("错误: 无法导入 asr_py 模块")
        print("请确保已设置 PYTHONPATH:")
        print("  export PYTHONPATH=/path/to/stt/build/python")
        print(f"\n详细错误: {e}")
        sys.exit(1)

    # 导入 evo_audio
    try:
        import evo_audio
        from evo_audio import AudioCapture
    except ImportError as e:
        print("错误: 无法导入 evo_audio 模块")
        print("请先编译 audio 组件:")
        print("  cd audio && mkdir -p build && cd build")
        print("  cmake .. && make -j$(nproc)")
        print("\n然后设置 PYTHONPATH:")
        print("  export PYTHONPATH=$PYTHONPATH:/path/to/audio/python")
        print(f"\n详细错误: {e}")
        sys.exit(1)

    return asr, evo_audio, AudioCapture


def list_devices(AudioCapture):
    """列出音频设备"""
    print("=== 音频输入设备 ===")
    for idx, name in AudioCapture.list_devices():
        print(f"  [{idx}] {name}")


def bytes_to_float(data: bytes, channels: int = 1) -> np.ndarray:
    """
    将 PCM S16LE 字节转换为 float32 数组

    Args:
        data: PCM S16LE 字节数据
        channels: 声道数

    Returns:
        float32 numpy 数组 (如果是多声道，会混音为 mono)
    """
    samples = len(data) // 2
    int16_array = np.frombuffer(data, dtype=np.int16)
    float_array = int16_array.astype(np.float32) / 32768.0

    # 多声道混音为 mono
    if channels > 1:
        frames = samples // channels
        float_array = float_array[:frames * channels].reshape(frames, channels)
        float_array = float_array.mean(axis=1)

    return float_array


def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """
    重采样音频

    Args:
        audio: 输入音频 (float32)
        src_rate: 源采样率
        dst_rate: 目标采样率

    Returns:
        重采样后的音频
    """
    if src_rate == dst_rate:
        return audio

    if HAS_SCIPY:
        # 使用 scipy 高质量重采样
        num_samples = int(len(audio) * dst_rate / src_rate)
        return scipy_signal.resample(audio, num_samples).astype(np.float32)
    else:
        # 简单线性插值
        ratio = dst_rate / src_rate
        num_samples = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, num_samples)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def run_stream_recognition(asr, evo_audio, AudioCapture, args):
    """运行流式识别"""
    # 配置 evo_audio
    evo_audio.init(
        sample_rate=16000,
        channels=1,
        chunk_size=3200,
        capture_device=args.device,
    )

    config = evo_audio.get_config()
    actual_rate = config['sample_rate']
    actual_channels = config['channels']

    print(f"音频配置: {actual_rate}Hz, {actual_channels}ch")

    # 检查是否需要重采样
    need_resample = (actual_rate != 16000)
    need_mixdown = (actual_channels > 1)

    if need_resample:
        method = "scipy" if HAS_SCIPY else "线性插值"
        print(f"  -> 重采样: {actual_rate}Hz -> 16000Hz ({method})")
        if not HAS_SCIPY:
            print("     提示: pip install scipy 可获得更好的重采样质量")

    if need_mixdown:
        print(f"  -> 混音: {actual_channels}ch -> 1ch")

    # 配置 ASR
    lang_map = {
        'zh': asr.Language.ZH,
        'en': asr.Language.EN,
        'ja': asr.Language.JA,
        'ko': asr.Language.KO,
        'yue': asr.Language.YUE,
        'auto': asr.Language.AUTO,
    }

    # 使用 ASRConfig（C++ 绑定的真实类名）
    asr_config = asr.ASRConfig.sensevoice(args.model_dir)
    asr_config.language = lang_map[args.language]
    asr_config.use_itn = True

    print(f"ASR 配置: 语言={args.language}, 每段时长={args.duration}秒")
    print("\n按 Ctrl+C 退出\n")

    # 音频缓冲
    audio_chunks = []

    def on_audio(data: bytes):
        audio_chunks.append(data)

    # 初始化引擎（使用 ASREngine）
    engine = asr.ASREngine()
    err = engine.initialize(asr_config)
    if not err.is_ok():
        print(f"错误: 初始化失败 - {err.message()}")
        sys.exit(1)

    try:
        round_num = 0

        try:
            while True:
                round_num += 1
                audio_chunks.clear()

                print(f"[轮次 {round_num}] 正在录音 ({args.duration}秒)...", end="", flush=True)

                with AudioCapture() as cap:
                    cap.set_callback(on_audio)
                    cap.start()
                    time.sleep(args.duration)

                audio_bytes = b''.join(audio_chunks)
                audio_float = bytes_to_float(audio_bytes, actual_channels)

                if need_resample:
                    audio_float = resample_audio(audio_float, actual_rate, 16000)

                print(f" {len(audio_float)} 样本 @ 16kHz")

                if len(audio_float) > 0:
                    result = engine.recognize(audio_float)

                    if result.text:
                        print(f">>> {result.text}")
                        print(f"    RTF: {result.rtf:.3f}\n")
                    else:
                        print(">>> (未检测到语音)\n")

        except KeyboardInterrupt:
            print("\n\n用户中断")

    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description='ASR 流式识别 (需要 evo_audio)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python stream_demo.py                 # 默认设备，持续识别
  python stream_demo.py -l              # 列出音频设备
  python stream_demo.py -d 0            # 使用设备 0
  python stream_demo.py --duration 5    # 每次录制 5 秒
  python stream_demo.py --language en   # 英文识别
        """
    )
    parser.add_argument('-l', '--list', action='store_true',
                        help='列出音频设备')
    parser.add_argument('-d', '--device', type=int, default=-1,
                        help='音频设备索引 (默认: -1 自动选择)')
    parser.add_argument('--duration', type=float, default=3.0,
                        help='每次录音时长秒数 (默认: 3.0)')
    parser.add_argument('--model-dir', '-m', default='~/.cache/sensevoice',
                        help='模型目录 (默认: ~/.cache/sensevoice)')
    parser.add_argument('--language', default='zh',
                        choices=['zh', 'en', 'ja', 'ko', 'yue', 'auto'],
                        help='识别语言 (默认: zh)')

    args = parser.parse_args()

    asr, evo_audio, AudioCapture = import_modules()

    print(f"ASR 版本: {asr.__version__}")
    print(f"EvoAudio 版本: {evo_audio.__version__}")

    if args.list:
        list_devices(AudioCapture)
        return

    run_stream_recognition(asr, evo_audio, AudioCapture, args)


if __name__ == "__main__":
    main()
