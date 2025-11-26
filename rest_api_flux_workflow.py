from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import json
import tempfile
import os
import time
import websocket
import threading
from typing import Optional
import base64
from io import BytesIO
from pathlib import Path

# Config
COMFY_API = "http://127.0.0.1:8188"
UPLOAD_ENDPOINT = f"{COMFY_API}/upload/image"
PROMPT_ENDPOINT = f"{COMFY_API}/prompt"
HISTORY_ENDPOINT = f"{COMFY_API}/history"
VIEW_ENDPOINT = f"{COMFY_API}/view"
WS_ENDPOINT = "ws://127.0.0.1:8188/ws"

# Workflow paths
WORKFLOW_DIR = Path("/WAN22") / "ComfyUI" / "user" / "default" / "workflows"
FLUX_WORKFLOW_PATH = "workflow/flux_workflow.json"

app = FastAPI()

# Allow CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Flux Schnell workflow template - will be loaded from file
FLUX_WORKFLOW_TEMPLATE = None

def load_workflow(workflow_path: Path) -> dict:
    """Load workflow from JSON file"""
    try:
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
        
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        # If it's a ComfyUI export format, extract the workflow
        if isinstance(workflow, dict) and "nodes" in workflow:
            # Convert ComfyUI node format to API format
            return convert_comfyui_workflow(workflow)
        
        return workflow
    except Exception as e:
        print(f"Error loading workflow from {workflow_path}: {e}")
        raise

def convert_comfyui_workflow(comfyui_workflow: dict) -> dict:
    """Convert ComfyUI node format to API format"""
    api_workflow = {}
    
    nodes = comfyui_workflow.get("nodes", [])
    links = comfyui_workflow.get("links", [])
    
    # List of non-executable node types that should be filtered out
    non_executable_nodes = {"Note", "Reroute"}
    
    # Create link mapping
    link_map = {}
    for link in links:
        link_id, output_node, output_slot, input_node, input_slot, data_type = link
        link_map[link_id] = {
            "output_node": output_node,
            "output_slot": output_slot,
            "input_node": input_node,
            "input_slot": input_slot
        }
    
    # Convert nodes
    for node in nodes:
        node_id = str(node["id"])
        node_type = node["type"]
        
        # Skip non-executable nodes like Note, Reroute, etc.
        if node_type in non_executable_nodes:
            continue
        
        api_node = {
            "class_type": node_type,
            "inputs": {}
        }
        
        # Preserve node title for identification
        if "title" in node:
            api_node["title"] = node["title"]
        
        # Add widget values as inputs
        if "widgets_values" in node and node["widgets_values"]:
            # Map widget values to input names based on node type
            widget_values = node["widgets_values"]
            
            # Common mappings for different node types
            if node_type == "CLIPTextEncode" and len(widget_values) > 0:
                api_node["inputs"]["text"] = widget_values[0]
            elif node_type == "KSampler" and len(widget_values) >= 7:
                # Handle KSampler with proper mapping: [seed, control, steps, cfg, sampler_name, scheduler, denoise]
                api_node["inputs"].update({
                    "seed": widget_values[0],
                    "control_after_generate": widget_values[1] if len(widget_values) > 1 else "randomize",
                    "steps": widget_values[2] if len(widget_values) > 2 else 4,
                    "cfg": widget_values[3] if len(widget_values) > 3 else 1.0,
                    "sampler_name": widget_values[4] if len(widget_values) > 4 else "euler",
                    "scheduler": widget_values[5] if len(widget_values) > 5 else "simple",
                    "denoise": widget_values[6] if len(widget_values) > 6 else 1.0
                })
            elif node_type == "EmptySD3LatentImage" and len(widget_values) >= 3:
                api_node["inputs"].update({
                    "width": widget_values[0],
                    "height": widget_values[1],
                    "batch_size": widget_values[2]
                })
            elif node_type == "CheckpointLoaderSimple" and len(widget_values) > 0:
                api_node["inputs"]["ckpt_name"] = widget_values[0]
            elif node_type == "SaveImage" and len(widget_values) > 0:
                api_node["inputs"]["filename_prefix"] = widget_values[0]
        
        # Add input connections
        if "inputs" in node:
            for input_info in node["inputs"]:
                if "link" in input_info and input_info["link"] is not None:
                    link_id = input_info["link"]
                    if link_id in link_map:
                        link_info = link_map[link_id]
                        input_name = input_info["name"]
                        api_node["inputs"][input_name] = [str(link_info["output_node"]), link_info["output_slot"]]
        
        api_workflow[node_id] = api_node
    
    return api_workflow

class ComfyUIMonitor:
    def __init__(self):
        self.completion_status = {}
        self.ws = None
        self.running = False
    
    def start_monitoring(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self._monitor_websocket, daemon=True).start()
    
    def _monitor_websocket(self):
        retry_count = 0
        max_retries = 5
        
        while retry_count < max_retries:
            try:
                print(f"Attempting WebSocket connection to {WS_ENDPOINT} (attempt {retry_count + 1})")
                self.ws = websocket.WebSocketApp(
                    WS_ENDPOINT,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open
                )
                self.ws.run_forever()
                break  # If we get here, connection was successful
            except Exception as e:
                print(f"WebSocket connection failed: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print("Max WebSocket retry attempts reached")
    
    def _on_open(self, ws):
        print("WebSocket connection established")
        self.running = True
    
    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            # Debug logging for WebSocket messages (can be commented out in production)
            # print(f"WebSocket message: {data}")
            
            if data.get("type") == "executing":
                prompt_id = data.get("data", {}).get("prompt_id")
                node = data.get("data", {}).get("node")
                
                if prompt_id:
                    if node is None:  # Execution completed
                        print(f"Execution completed for prompt {prompt_id}")
                        self.completion_status[prompt_id] = "completed"
                    else:
                        print(f"Executing node {node} for prompt {prompt_id}")
                        self.completion_status[prompt_id] = "running"
            
            # Check for progress_state messages that show all nodes finished
            elif data.get("type") == "progress_state":
                prompt_id = data.get("data", {}).get("prompt_id")
                nodes = data.get("data", {}).get("nodes", {})
                if prompt_id and nodes:
                    all_finished = all(node.get("state") == "finished" for node in nodes.values())
                    if all_finished:
                        print(f"All nodes finished for prompt {prompt_id} (via progress_state)")
                        self.completion_status[prompt_id] = "completed"
            
            # Also check for 'executed' type which might indicate completion
            elif data.get("type") == "executed":
                prompt_id = data.get("data", {}).get("prompt_id")
                if prompt_id:
                    print(f"Node executed for prompt {prompt_id}")
                    # Don't mark as completed here, wait for the executing with node=None
            
            # Check for execution_error or other error types
            elif data.get("type") in ["execution_error", "execution_interrupted"]:
                prompt_id = data.get("data", {}).get("prompt_id")
                if prompt_id:
                    print(f"Execution error/interrupted for prompt {prompt_id}")
                    self.completion_status[prompt_id] = "error"
                    
        except Exception as e:
            print(f"WebSocket message error: {e}")
            print(f"Raw message: {message}")
    
    def _on_error(self, ws, error):
        print(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        print("WebSocket connection closed")
        self.running = False
    
    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> bool:
        """Wait for a prompt to complete execution"""
        start_time = time.time()
        last_check_time = 0
        
        while time.time() - start_time < timeout:
            current_time = time.time()
            
            # Check WebSocket status
            if prompt_id in self.completion_status:
                status = self.completion_status[prompt_id]
                if status == "completed":
                    print(f"Prompt {prompt_id} completed via WebSocket")
                    return True
                elif status == "error":
                    print(f"Prompt {prompt_id} failed via WebSocket")
                    return False
            
            # Fallback: Check ComfyUI history API every 5 seconds if WebSocket fails
            if current_time - last_check_time > 5:
                try:
                    response = requests.get(f"{HISTORY_ENDPOINT}/{prompt_id}", timeout=30)
                    if response.status_code == 200:
                        history = response.json()
                        if prompt_id in history:
                            # Check if there are outputs - this means completion
                            outputs = history[prompt_id].get("outputs", {})
                            if outputs:
                                print(f"Prompt {prompt_id} completed via history API fallback")
                                self.completion_status[prompt_id] = "completed"
                                return True
                            
                            # Check for status indicators
                            status = history[prompt_id].get("status", {})
                            if status.get("completed", False):
                                print(f"Prompt {prompt_id} completed via status check")
                                self.completion_status[prompt_id] = "completed"
                                return True
                except Exception as e:
                    print(f"History API check failed: {e}")
                
                last_check_time = current_time
            
            time.sleep(0.5)
        
        print(f"Timeout waiting for prompt {prompt_id}")
        return False

# Global monitor instance
monitor = ComfyUIMonitor()

@app.on_event("startup")
async def startup_event():
    monitor.start_monitoring()
    # Load workflows on startup
    global FLUX_WORKFLOW_TEMPLATE
    try:
        FLUX_WORKFLOW_TEMPLATE = load_workflow(Path(FLUX_WORKFLOW_PATH))
        print(f"Loaded Flux workflow from {FLUX_WORKFLOW_PATH}")
    except Exception as e:
        print(f"Warning: Could not load Flux workflow: {e}")
        FLUX_WORKFLOW_TEMPLATE = None

def find_text_encode_nodes(workflow: dict) -> tuple:
    """Find positive and negative text encode nodes in workflow"""
    positive_node = None
    negative_node = None
    
    for node_id, node in workflow.items():
        if node.get("class_type") == "CLIPTextEncode":
            # Check node title first (most reliable)
            title = node.get("title", "")
            if "Positive" in title:
                positive_node = node_id
            elif "Negative" in title:
                negative_node = node_id
            else:
                # Fallback: check text length - longer text is likely positive
                text_input = node.get("inputs", {}).get("text", "")
                if len(text_input) > 10:  # Assume longer text is positive
                    if positive_node is None:
                        positive_node = node_id
                elif text_input == "":  # Empty text is likely negative
                    if negative_node is None:
                        negative_node = node_id
    
    return positive_node, negative_node

def find_latent_image_node(workflow: dict) -> str:
    """Find the latent image generation node"""
    for node_id, node in workflow.items():
        if node.get("class_type") in ["EmptyLatentImage", "EmptySD3LatentImage"]:
            return node_id
    return None

def find_sampler_node(workflow: dict) -> str:
    """Find the sampler/scheduler node"""
    for node_id, node in workflow.items():
        if node.get("class_type") in ["KSampler", "KSamplerAdvanced"]:
            return node_id
    return None

def get_generated_images(prompt_id: str) -> list:
    """Get the generated images from ComfyUI history"""
    try:
        print(f"Fetching history for prompt_id: {prompt_id}")
        response = requests.get(f"{HISTORY_ENDPOINT}/{prompt_id}")
        response.raise_for_status()
        history = response.json()
        
        print(f"History response: {history}")
        
        if prompt_id in history:
            prompt_data = history[prompt_id]
            print(f"Prompt data keys: {list(prompt_data.keys())}")
            
            outputs = prompt_data.get("outputs", {})
            print(f"Outputs: {outputs}")
            
            images = []
            
            for node_id, output in outputs.items():
                print(f"Processing node {node_id}: {output}")
                if "images" in output:
                    for img_info in output["images"]:
                        print(f"Found image: {img_info}")
                        images.append({
                            "filename": img_info["filename"],
                            "subfolder": img_info.get("subfolder", ""),
                            "type": img_info.get("type", "output")
                        })
                else:
                    print(f"No 'images' key in output for node {node_id}")
            
            print(f"Total images found: {len(images)}")
            return images
        else:
            print(f"Prompt ID {prompt_id} not found in history")
            print(f"Available prompt IDs: {list(history.keys())}")
    except Exception as e:
        print(f"Error getting images: {e}")
        import traceback
        traceback.print_exc()
    return []



@app.post("/flux/text2image")
async def flux_text2image(
    prompt: str = Form(...),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(4),
    seed: Optional[int] = Form(None),
    return_base64: bool = Form(False)
):
    """
    Generate an image using Flux model from loaded workflow
    """
    try:
        if FLUX_WORKFLOW_TEMPLATE is None:
            raise HTTPException(status_code=500, detail="Flux workflow not loaded. Check if flux_workflow.json exists in user/default/workflows/")
        
        # Create a copy of the workflow
        workflow = json.loads(json.dumps(FLUX_WORKFLOW_TEMPLATE))
        
        # Find relevant nodes dynamically
        positive_node, negative_node = find_text_encode_nodes(workflow)
        latent_node = find_latent_image_node(workflow)
        sampler_node = find_sampler_node(workflow)
        
        print(f"Found nodes - Positive: {positive_node}, Negative: {negative_node}, Latent: {latent_node}, Sampler: {sampler_node}")
        
        if not positive_node:
            raise HTTPException(status_code=500, detail="Could not find positive text encode node in workflow")
        
        # Update prompt
        print(f"Setting prompt '{prompt}' to node {positive_node}")
        workflow[positive_node]["inputs"]["text"] = prompt
        
        # Update image dimensions if latent node found
        if latent_node:
            workflow[latent_node]["inputs"]["width"] = width
            workflow[latent_node]["inputs"]["height"] = height
        
        # Update sampling parameters if sampler node found
        if sampler_node:
            workflow[sampler_node]["inputs"]["steps"] = max(1, min(steps, 10))
            if seed is not None:
                workflow[sampler_node]["inputs"]["seed"] = seed
            else:
                import random
                workflow[sampler_node]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
            
            # Ensure control_after_generate is set properly
            if "control_after_generate" not in workflow[sampler_node]["inputs"]:
                workflow[sampler_node]["inputs"]["control_after_generate"] = "randomize"
        
        # Submit to ComfyUI
        response = requests.post(PROMPT_ENDPOINT, json={"prompt": workflow})
        response.raise_for_status()
        
        result = response.json()
        prompt_id = result.get("prompt_id")
        
        if not prompt_id:
            raise HTTPException(status_code=500, detail="Failed to get prompt ID from ComfyUI")
        
        # Wait for completion
        print(f"Waiting for completion of prompt {prompt_id}...")
        completed = monitor.wait_for_completion(prompt_id, timeout=300)
        
        if not completed:
            raise HTTPException(status_code=408, detail="Image generation timed out")
        
        # Get generated images
        print(f"Fetching generated images for prompt {prompt_id}...")
        images = get_generated_images(prompt_id)
        
        if not images:
            # Try multiple times with increasing delays
            for retry in range(3):
                delay = (retry + 1) * 2  # 2, 4, 6 seconds
                print(f"No images found, waiting {delay} seconds and retrying (attempt {retry + 1}/3)...")
                time.sleep(delay)
                images = get_generated_images(prompt_id)
                if images:
                    break
        
        if not images:
            raise HTTPException(status_code=500, detail="No images generated")
        
        # Get the first image
        img_info = images[0]
        print(f"Found image: {img_info}")
        
        # Construct image URL
        params = {
            "filename": img_info["filename"],
            "type": img_info["type"]
        }
        if img_info["subfolder"]:
            params["subfolder"] = img_info["subfolder"]
        
        print(f"Fetching image from ComfyUI with params: {params}")
        # Get image data
        img_response = requests.get(VIEW_ENDPOINT, params=params, timeout=60)
        img_response.raise_for_status()
        print(f"Successfully fetched image, size: {len(img_response.content)} bytes")
        
        if return_base64:
            # Return base64 encoded image
            img_base64 = base64.b64encode(img_response.content).decode('utf-8')
            return JSONResponse(content={
                "status": "success",
                "prompt_id": prompt_id,
                "image_base64": img_base64,
                "image_info": img_info
            })
        else:
            # Save image temporarily and return file
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    tmp_file.write(img_response.content)
                    tmp_file_path = tmp_file.name
                
                print(f"Saved temporary file: {tmp_file_path}")
                
                # Create a proper cleanup function
                async def cleanup_temp_file():
                    try:
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)
                            print(f"Cleaned up temporary file: {tmp_file_path}")
                    except Exception as e:
                        print(f"Error cleaning up temp file: {e}")
                
                return FileResponse(
                    path=tmp_file_path,
                    media_type="image/png",
                    filename=f"flux_generated_{prompt_id}.png",
                    background=cleanup_temp_file
                )
            except Exception as e:
                print(f"Error saving temporary file: {e}")
                raise HTTPException(status_code=500, detail=f"Error saving image file: {str(e)}")
    
    except requests.RequestException as e:
        print(f"ComfyUI API error in flux/text2image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ComfyUI API error: {str(e)}")
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Internal error in flux/text2image: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if ComfyUI is running
        response = requests.get(f"{COMFY_API}/system_stats", timeout=5)
        comfy_status = "running" if response.status_code == 200 else "down"
    except:
        comfy_status = "down"
    
    return {
        "status": "healthy",
        "comfyui_status": comfy_status,
        "websocket_monitoring": monitor.running
    }

# Run with: uvicorn rest_api_workflow:app --reload
if __name__ == "__main__":
    uvicorn.run("rest_api_flux_workflow:app", host="0.0.0.0", port=9000, reload=True)