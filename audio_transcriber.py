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
        print(f"[OK] 模型加载成功 (device={device})")


    def transcribe(self, audio_path, language) -> List[Dict[str, Any]]:
        """
        尝试使用 CUDA 转录，如果失败则回退到 CPU。
        包含详细的错误处理和内存清理。
        """
        import sys
        
        lang = language
        if lang == "zh-CN" or lang == "tw":
            lang = "zh"

        # 用 splitext 处理扩展名，兼容含中文、含多个点的文件名（如：xxx_《标题》.mp3）
        root, _ = os.path.splitext(audio_path)
        transcribe_file = root + ".srt.json"
        
        try:
            print(f"[INFO] 开始转录 (language={lang})...")
            
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
            print(f"[INFO] 音频信息: language={info.language}, duration={info.duration:.1f}s")
            
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
            
            print(f"[OK] 转录完成，共 {len(srt_segments)} 个片段")
            srt_segments = self.merge_sentences(srt_segments, 3, 22)
            with open(transcribe_file, "w", encoding="utf-8") as f:
                json.dump(srt_segments, f, ensure_ascii=False, indent=2)
            
            return srt_segments, transcribe_file
            
        except Exception as e:
            print(f"[ERROR] 转录失败: {type(e).__name__}: {e}")
            
        return [], ""



    def merge_sentences(self, input_segments, min_sentence_duration, max_sentence_duration):
        print(f"[INFO] 合并句子...")
        i = 0
        while i < len(input_segments):
            if i+1 < len(input_segments):
                if input_segments[i]['end'] - input_segments[i]['start'] > max_sentence_duration:
                    i += 1
                    continue
                if input_segments[i]['end'] - input_segments[i]['start'] < min_sentence_duration or input_segments[i+1]['end'] - input_segments[i+1]['start'] < min_sentence_duration:
                    input_segments[i]['caption'] += input_segments[i+1]['caption']
                    input_segments[i]['end'] = input_segments[i+1]['end']
                    input_segments.pop(i+1)
                    # 不递增 i，因为 result[i] 现在是合并后的元素，可能需要继续与下一个元素合并
                else:
                    i += 1
            else:
                i += 1

        final_segments = []
        for seg in input_segments:
            final_segments.append({
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "duration": float(seg["end"]) - float(seg["start"]),
                "caption": seg["caption"]
            })

        if len(final_segments) > 0:
            if final_segments[0]['start'] != 0.0:
                final_segments[0]['start'] = 0.0

            end_time = 0.0
            for seg in final_segments:
                if end_time > 0 and seg['start'] != end_time: # not the 1st item
                    seg['start'] = end_time # fix start time (must == end of last item)
                seg['duration'] = seg['end'] - seg['start']
                end_time = seg['end']

        print(f"[OK] 合并完成，共 {len(final_segments)} 个片段")
        return final_segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="音频转文字 (Whisper)")
    parser.add_argument("audio_path", help="音频文件路径 (如 .mp3 / .wav)")
    parser.add_argument("-l", "--language", default="zh", help="语言代码 (默认: zh)")
    args = parser.parse_args()
    transcriber = AudioTranscriber("small", "cuda", "float16")
    srt_segments, srt_file = transcriber.transcribe(args.audio_path, args.language)
    # save to srt file（与 transcribe 内一致：用 splitext 支持含中文等复杂文件名）
