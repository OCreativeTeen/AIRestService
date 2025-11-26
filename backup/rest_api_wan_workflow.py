from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import json
import tempfile
import os
from pathlib import Path

# Config
COMFY_API = "http://127.0.0.1:8188"
UPLOAD_ENDPOINT = f"{COMFY_API}/upload/image"
PROMPT_ENDPOINT = f"{COMFY_API}/prompt"

# Workflow paths
WORKFLOW_DIR = Path("/WAN22") / "ComfyUI" / "user" / "default" / "workflows"
WAN_WORKFLOW_PATH = WORKFLOW_DIR / "wan_workflow.json"

app = FastAPI()

# Allow CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/wan/image2video")
async def run_workflow(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        # Save image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as tmp_img:
            tmp_img.write(await image.read())
            tmp_img_path = tmp_img.name

        # Upload image to ComfyUI
        with open(tmp_img_path, "rb") as f:
            res = requests.post(UPLOAD_ENDPOINT, files={"image": f})
            res.raise_for_status()
            image_name = res.json()["name"]

        # Load workflow from filesystem
        with open(WAN_WORKFLOW_PATH, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)
        
        # Update prompt + image
        workflow_data["52"]["widgets_values"][0] = image_name
        workflow_data["6"]["widgets_values"][0] = prompt

        # Send modified workflow to ComfyUI
        response = requests.post(PROMPT_ENDPOINT, json={"prompt": workflow_data})
        response.raise_for_status()

        return JSONResponse(content={"status": "submitted", "response": response.json()}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        # Clean up temp image
        if os.path.exists(tmp_img_path):
            os.remove(tmp_img_path)


# Run with: uvicorn rest_api_wan_workflow:app --reload
if __name__ == "__main__":
    uvicorn.run("rest_api_wan_workflow:app", host="0.0.0.0", port=9000, reload=True)
