from fastapi import FastAPI, File, UploadFile, Form, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import os
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
