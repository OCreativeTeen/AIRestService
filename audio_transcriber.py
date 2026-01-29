from faster_whisper import WhisperModel
from datetime import datetime
import re
import os
import sys
import json
import argparse
from difflib import SequenceMatcher
from typing import List, Dict, Any
import gc

class AudioTranscriber:

    def __init__(self, model_size, device, compute_type):
        #device = "cuda"
        #compute_type = "int8"
        self.model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type,
            cpu_threads=4 if device == "cpu" else 1,
            num_workers=1
        )
        print(f"✅ 模型加载成功 (device={device})")


    def transcribe(self, audio_path, language) -> List[Dict[str, Any]]:
        """
        尝试使用 CUDA 转录，如果失败则回退到 CPU。
        包含详细的错误处理和内存清理。
        """
        import sys
        
        lang = language
        if lang == "zh-CN" or lang == "tw":
            lang = "zh"

        if audio_path.endswith('.mp3'):
            transcribe_file = audio_path.replace(".mp3", ".srt.json")        
        elif audio_path.endswith('.wav'):
            transcribe_file = audio_path.replace(".wav", ".srt.json")
        else:
            transcribe_file = audio_path + ".srt.json"
        if os.path.exists(transcribe_file):
            with open(transcribe_file, "r", encoding="utf-8") as f:
                segments = json.load(f)
            return segments

        
        model = None
        try:
            print(f"📝 开始转录 (language={lang})...")
            
            # 使用低内存设置转录
            segments_gen, info = self.model.transcribe(
                audio_path, 
                beam_size=1,  # 最小 beam_size
                language=lang,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                condition_on_previous_text=False,
                word_timestamps=False,  # 禁用词级时间戳，节省内存
            )
            print(f"📝 音频信息: language={info.language}, duration={info.duration:.1f}s")
            
            # 迭代生成器
            srt_segments = []
            seg_count = 0
            for seg in segments_gen:
                seg_count += 1
                if seg_count % 10 == 0:
                    print(f"   处理片段 {seg_count}...")
                    sys.stdout.flush()
                srt_segments.append({
                    'start': seg.start,
                    'end': seg.end,
                    'caption': seg.text
                })
            
            print(f"✅ 转录完成，共 {len(srt_segments)} 个片段")
            with open(transcribe_file, "w", encoding="utf-8") as f:
                json.dump(srt_segments, f, ensure_ascii=False, indent=2)
            return srt_segments
            
        except Exception as e:
            print(f"❌ 使用 失败: {type(e).__name__}: {e}")
            
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="音频转文字 (Whisper)")
    parser.add_argument("audio_path", help="音频文件路径 (如 .mp3 / .wav)")
    parser.add_argument("-l", "--language", default="zh", help="语言代码 (默认: zh)")
    args = parser.parse_args()
    transcriber = AudioTranscriber("small", "cuda", "float16")
    srt_segments = transcriber.transcribe(args.audio_path, args.language)
    # save to srt file
    with open(args.audio_path.replace(".mp3", ".srt").replace(".wav", ".srt"), "w", encoding="utf-8") as f:
        for seg in srt_segments:
            f.write(f"{seg['start']} --> {seg['end']}\n{seg['caption']}\n\n")
    print(f"✅ 转录完成，共 {len(srt_segments)} 个片段")