from fastapi import FastAPI, File, UploadFile, Form, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import os
import sys
import subprocess
import tempfile
from typing import Dict, List
from workflow_processor import WorkflowProcessor, WorkflowConfig

# Config
COMFY_API = "http://127.0.0.1:8188"
UPLOAD_IMAGE_ENDPOINT = f"{COMFY_API}/upload/image"
COMFYUI_INPUT_FILE_FOLDER = "O:/input"
PROMPT_ENDPOINT = f"{COMFY_API}/prompt"

app = FastAPI()

# Allow CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize workflow processor
processor = WorkflowProcessor(
    comfy_api=COMFY_API,
    upload_endpoint=UPLOAD_IMAGE_ENDPOINT,
    prompt_endpoint=PROMPT_ENDPOINT,
    input_folder=COMFYUI_INPUT_FILE_FOLDER
)

# Load configurations
def load_all_configs():
    """Load all workflow configurations from config directory"""
    configs = {}
    config_dir = "config"
    if os.path.exists(config_dir):
        for filename in os.listdir(config_dir):
            if filename.endswith('_config.json'):
                config_path = os.path.join(config_dir, filename)
                try:
                    config = processor.load_workflow_config(config_path)
                    configs[config.endpoint_path] = config
                except Exception as e:
                    print(f"Error loading config {filename}: {e}")
    return configs

# Load configurations at startup
workflow_configs = load_all_configs()


@app.post("/wan/image2video")
async def run_workflow(
    first_frame: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(...),
    filename_prefix: str = Form(...),
    image_width: int = Form(...),
    image_height: int = Form(...),    
    cfg_scale: float = Form(...),
    steps: int = Form(...),
    seed: int = Form(...),
    motion_frame: int = Form(...),
    frame_rate: int = Form(...),
    num_frames: int = Form(...)
):
    """Legacy image2video endpoint"""
    if "/wan/image2video" in workflow_configs:
        config = workflow_configs["/wan/image2video"]
        return await processor.execute_workflow(
            config, 
            { "prompt": prompt, "negative_prompt": negative_prompt,  "filename_prefix": filename_prefix,
              "image_width": image_width, "image_height": image_height, 
              "cfg_scale": cfg_scale, "steps": steps, "seed": seed, 
              "motion_frame": motion_frame, "frame_rate": frame_rate, "num_frames": num_frames },
            { "first_frame": first_frame }
        )


@app.post("/wan/imagesss2video")
async def run_workflow(
    first_frame: UploadFile = File(...),
    last_frame: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(...),
    filename_prefix: str = Form(...),
    image_width: int = Form(...),
    image_height: int = Form(...),    
    cfg_scale: float = Form(...),
    steps: int = Form(...),
    seed: int = Form(...),
    motion_frame: int = Form(...),
    frame_rate: int = Form(...),
    num_frames: int = Form(...)
):
    if "/wan/imagesss2video" in workflow_configs:
        config = workflow_configs["/wan/imagesss2video"]
        return await processor.execute_workflow(
            config, 
            { "prompt": prompt, "negative_prompt": negative_prompt,  "filename_prefix": filename_prefix,
              "image_width": image_width, "image_height": image_height, 
              "cfg_scale": cfg_scale, "steps": steps, "seed": seed, 
              "motion_frame": motion_frame, "frame_rate": frame_rate, "num_frames": num_frames },
            { "first_frame": first_frame, "last_frame": last_frame }
        )


@app.post("/wan/s2v")
async def run_workflow(
    image: UploadFile = File(...),
    sound: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(...),
    filename_prefix: str = Form(...),
    image_width: int = Form(...),
    image_height: int = Form(...),    
    cfg_scale: float = Form(...),
    steps: int = Form(...),
    seed: int = Form(...),
    motion_frame: int = Form(...),
    frame_rate: int = Form(...),
    num_frames: int = Form(...)
):
    if "/wan/s2v" in workflow_configs:
        config = workflow_configs["/wan/s2v"]
        return await processor.execute_workflow(
            config, 
            { "prompt": prompt, "negative_prompt": negative_prompt,  "filename_prefix": filename_prefix,
              "image_width": image_width, "image_height": image_height, 
              "cfg_scale": cfg_scale, "steps": steps, "seed": seed, 
              "motion_frame": motion_frame, "frame_rate": frame_rate, "num_frames": num_frames },
            { "image": image, "audio": sound }
        )


@app.post("/wan/infinite_s2v")
async def run_sound2video_workflow(
    image: UploadFile = File(...),
    sound: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    filename_prefix: str = Form("infinite_s2v"),
    image_width: int = Form(1024),
    image_height: int = Form(576),
    cfg_scale: float = Form(7.5),
    steps: int = Form(20),
    seed: int = Form(0),
    motion_frame: int = Form(16),
    frame_rate: int = Form(24),
    num_frames: int = Form(121),
):
    """Legacy sound2video endpoint"""
    if "/wan/infinite_s2v" in workflow_configs:
        config = workflow_configs["/wan/infinite_s2v"]
        return await processor.execute_workflow(
            config, 
            { "prompt": prompt, "negative_prompt": negative_prompt,  "filename_prefix": filename_prefix,
              "image_width": image_width, "image_height": image_height, 
              "cfg_scale": cfg_scale, "steps": steps, "seed": seed, 
              "motion_frame": motion_frame, "frame_rate": frame_rate, "num_frames": num_frames },
            { "image": image, "audio": sound }
        )


@app.post("/wan/action_transfer")
async def run_actiontransfer_workflow(
    image: UploadFile = File(...),
    action: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(...),
    filename_prefix: str = Form(...),
    image_width: int = Form(...),
    image_height: int = Form(...),    
    cfg_scale: float = Form(...),
    steps: int = Form(...),
    seed: int = Form(...),
    motion_frame: int = Form(...),
    frame_rate: int = Form(...),
    num_frames: int = Form(...)
):
    if "/wan/action_transfer" in workflow_configs:
        config = workflow_configs["/wan/action_transfer"]
        return await processor.execute_workflow(
            config, 
            { "prompt": prompt, "negative_prompt": negative_prompt,  "filename_prefix": filename_prefix,
              "image_width": image_width, "image_height": image_height, 
              "cfg_scale": cfg_scale, "steps": steps, "seed": seed, 
              "motion_frame": motion_frame, "frame_rate": frame_rate, "num_frames": num_frames },
            { "image": image, "action": action }
        )



@app.post("/interpolate")
async def run_interpolate_workflow(
    video: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(...),
    filename_prefix: str = Form(...),
    image_width: int = Form(...),
    image_height: int = Form(...),    
    cfg_scale: float = Form(...),
    steps: int = Form(...),
    seed: int = Form(...),
    motion_frame: int = Form(...),
    frame_rate: int = Form(...),
    num_frames: int = Form(...)
):
    if "/interpolate" in workflow_configs:
        config = workflow_configs["/interpolate"]
        return await processor.execute_workflow(
            config, 
            { "prompt": prompt, "negative_prompt": negative_prompt,  "filename_prefix": filename_prefix,
              "image_width": image_width, "image_height": image_height, 
              "cfg_scale": cfg_scale, "steps": steps, "seed": seed, 
              "motion_frame": motion_frame, "frame_rate": frame_rate, "num_frames": num_frames },
            { "video": video }
        )


#
# 只传音频（默认语言 zh）：
# curl -X POST "http://localhost:9001/transcribe" -F "audio_file=@/path/to/your/audio.mp3"
#
# 指定语言（例如英文）：
# curl -X POST "http://localhost:9001/transcribe" -F "audio_file=@/path/to/your/audio.mp3" -F "language=en"
#
# Window
# curl -X POST "http://localhost:9001/transcribe" -F "audio_file=@E:/ComfyUI-Easy/AIRestService/11.mp3" -F "language=zh"
@app.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    language: str = Form("zh"),
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ext = os.path.splitext(audio_file.filename or "audio.mp3")[1] or ".mp3"
    fd, temp_path = tempfile.mkstemp(suffix=ext, prefix="transcribe_", dir=script_dir)
    if temp_path.endswith(".mp3") or temp_path.endswith(".wav"):
        root, _ = os.path.splitext(temp_path)
        srt_json_path = root + ".srt.json"
    else:
        return JSONResponse(
            content={"error": "Bad audio file", "stderr": proc.stderr or proc.stdout},
            status_code=500,
        )

    try:
        os.close(fd)
        with open(temp_path, "wb") as f:
            f.write(await audio_file.read())
        proc = subprocess.run(
            [sys.executable, os.path.join(script_dir, "audio_transcriber.py"), temp_path, "-l", language],
            cwd=script_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if not os.path.exists(srt_json_path):
            return JSONResponse(
                content={"error": "Transcriber did not produce output", "stderr": proc.stderr or ""},
                status_code=500,
            )

        with open(srt_json_path, "r", encoding="utf-8") as f:
            srt_segments = json.load(f)
        return srt_segments
        
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        if os.path.exists(srt_json_path):
            try:
                os.unlink(srt_json_path)
            except OSError:
                pass


# Generic workflow endpoint
@app.post("/{workflow_name}")
async def execute_workflow_by_name(
    workflow_name: str,
    prompt: str = Form(None),
    image: UploadFile = File(None),
    sound: UploadFile = File(None),
    video: UploadFile = File(None)
):
    """Generic workflow executor based on configuration"""
    endpoint_path = f"/{workflow_name}"
    
    if endpoint_path not in workflow_configs:
        return JSONResponse(
            content={"error": f"Workflow '{workflow_name}' not found"}, 
            status_code=404
        )
    
    config = workflow_configs[endpoint_path]
    
    form_data = {}
    return await processor.execute_workflow(config, form_data, uploaded_files)


# API to list available workflows
@app.get("/workflows")
async def list_workflows():
    """List all available workflow configurations"""
    return {
        "workflows": [
            {
                "endpoint": endpoint,
                "description": config.description,
                "form_params": config.form_params,
                "required_files": [fc.param_name for fc in config.file_configs if fc.required]
            }
            for endpoint, config in workflow_configs.items()
        ]
    }


# API to reload configurations
@app.post("/reload-config")
async def reload_configurations():
    """Reload all workflow configurations"""
    global workflow_configs
    try:
        workflow_configs = load_all_configs()
        return {"status": "success", "loaded_workflows": len(workflow_configs)}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



if __name__ == "__main__":
    uvicorn.run("rest_api_wan_workflow:app", host="0.0.0.0", port=9001, reload=True)
