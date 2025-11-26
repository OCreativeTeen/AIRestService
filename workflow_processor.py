from fastapi import File, UploadFile, Form
from fastapi.responses import JSONResponse
import json
import tempfile
import os
import shutil
import requests
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum


class FileType(Enum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class FileConfig:
    """Configuration for file upload and processing"""
    param_name: str  # FastAPI parameter name
    file_type: FileType
    upload_to_comfy: bool = False  # Whether to upload to ComfyUI
    copy_to_input: bool = False   # Whether to copy to ComfyUI input folder
    required: bool = True


@dataclass
class NodeUpdate:
    """Configuration for updating workflow nodes"""
    node_id: str
    input_key: str
    value_source: str  # "form_param", "uploaded_file_name", or "static_value"
    static_value: Optional[Any] = None
    map_key: Optional[str] = None  # Alternative parameter name for form mapping


@dataclass
class WorkflowConfig:
    """Complete workflow configuration"""
    endpoint_path: str
    workflow_file: str
    description: str
    form_params: List[str]  # Form parameter names
    file_configs: List[FileConfig]
    node_updates: List[NodeUpdate]


class WorkflowProcessor:
    """Generic workflow processor that handles file uploads and node updates"""
    
    def __init__(self, comfy_api: str, upload_endpoint: str, 
                 prompt_endpoint: str, input_folder: str):
        self.comfy_api = comfy_api
        self.upload_endpoint = upload_endpoint
        self.prompt_endpoint = prompt_endpoint
        self.input_folder = input_folder
        
    def load_workflow_config(self, config_path: str) -> WorkflowConfig:
        """Load workflow configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                
            # Convert to WorkflowConfig object
            file_configs = [
                FileConfig(
                    param_name=fc['param_name'],
                    file_type=FileType(fc['file_type']),
                    upload_to_comfy=fc.get('upload_to_comfy', False),
                    copy_to_input=fc.get('copy_to_input', False),
                    required=fc.get('required', True)
                )
                for fc in config_data['file_configs']
            ]
            
            node_updates = [
                NodeUpdate(
                    node_id=nu['node_id'],
                    input_key=nu['input_key'],
                    value_source=nu['value_source'],
                    static_value=nu.get('static_value'),
                    map_key=nu.get('map_key')
                )
                for nu in config_data['node_updates']
            ]
            
            return WorkflowConfig(
                endpoint_path=config_data['endpoint_path'],
                workflow_file=config_data['workflow_file'],
                description=config_data['description'],
                form_params=config_data['form_params'],
                file_configs=file_configs,
                node_updates=node_updates
            )
        except Exception as e:
            raise Exception(f"Failed to load workflow config from {config_path}: {str(e)}")
    
    def load_workflow(self, workflow_path: str) -> Dict:
        """Load workflow JSON file"""
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise Exception(f"Workflow file not found at {workflow_path}")
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON in workflow file {workflow_path}")
    
    async def process_files(self, file_configs: List[FileConfig], 
                          uploaded_files: Dict[str, UploadFile]) -> Dict[str, str]:
        """Process uploaded files according to configuration"""
        temp_files = []
        processed_files = {}
        
        try:
            for file_config in file_configs:
                if file_config.param_name not in uploaded_files:
                    if file_config.required:
                        raise Exception(f"Required file {file_config.param_name} not provided")
                    continue
                
                uploaded_file = uploaded_files[file_config.param_name]
                
                # Save file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=os.path.splitext(uploaded_file.filename)[1]
                ) as tmp_file:
                    tmp_file.write(await uploaded_file.read())
                    tmp_file_path = tmp_file.name
                    temp_files.append(tmp_file_path)
                
                # Process based on configuration
                if file_config.upload_to_comfy:
                    # Upload to ComfyUI
                    with open(tmp_file_path, "rb") as f:
                        res = requests.post(self.upload_endpoint, files={"image": f})
                        res.raise_for_status()
                        processed_files[file_config.param_name] = res.json()["name"]
                
                if file_config.copy_to_input:
                    # Copy to ComfyUI input folder
                    target_path = os.path.join(self.input_folder, uploaded_file.filename)
                    shutil.copy(tmp_file_path, target_path)
                    processed_files[file_config.param_name] = uploaded_file.filename
                
                # If neither upload nor copy, just store the filename
                if not file_config.upload_to_comfy and not file_config.copy_to_input:
                    processed_files[file_config.param_name] = uploaded_file.filename
                    
        except Exception as e:
            # Clean up temp files on error
            for tmp_path in temp_files:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            raise e
        
        # Clean up temp files after successful processing
        for tmp_path in temp_files:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        return processed_files
    
    def update_workflow_nodes(self, workflow_data: Dict, node_updates: List[NodeUpdate],
                            form_data: Dict[str, str], processed_files: Dict[str, str]) -> Dict:
        """Update workflow nodes according to configuration"""
        for update in node_updates:
            if update.node_id not in workflow_data:
                continue
                
            # Ensure inputs dict exists
            if "inputs" not in workflow_data[update.node_id]:
                workflow_data[update.node_id]["inputs"] = {}
            
            # Determine the value to set
            value = None
            if update.value_source == "form_param":
                # Get value from form parameters
                # Priority 1: Try exact match with map_key if provided
                if update.map_key and update.map_key in form_data:
                    value = form_data[update.map_key]
                # Priority 2: Try exact match with input_key
                elif update.input_key in form_data:
                    value = form_data[update.input_key]
                # Priority 3: Try substring matching (fallback for legacy configs)
                else:
                    for param_name in form_data:
                        if update.map_key and (param_name in update.map_key or update.map_key in param_name):
                            value = form_data[param_name]
                            break
                        elif param_name in update.input_key or update.input_key in param_name:
                            value = form_data[param_name]
                            break
                    
            elif update.value_source == "uploaded_file_name":
                # Get filename from processed files
                for file_param in processed_files:
                    if update.map_key and (file_param in update.map_key or update.map_key in file_param):
                        value = processed_files[file_param]
                        break
                    if file_param in update.input_key or update.input_key in file_param:
                        value = processed_files[file_param]
                        break
                        
            elif update.value_source == "static_value":
                value = update.static_value
            
            # Set the value if found
            if value is not None:
                workflow_data[update.node_id]["inputs"][update.input_key] = value
                
        return workflow_data
    
    async def execute_workflow(self, config: WorkflowConfig, 
                             form_data: Dict[str, str], 
                             uploaded_files: Dict[str, UploadFile]) -> JSONResponse:
        """Execute workflow based on configuration"""
        try:
            # Process uploaded files
            processed_files = await self.process_files(config.file_configs, uploaded_files)
            
            # Load and update workflow
            workflow_data = self.load_workflow(config.workflow_file)
            workflow_data = self.update_workflow_nodes(
                workflow_data, config.node_updates, form_data, processed_files
            )
            
            # Send to ComfyUI
            response = requests.post(self.prompt_endpoint, json={"prompt": workflow_data})
            response.raise_for_status()
            
            return JSONResponse(
                content={"status": "submitted", "response": response.json()}, 
                status_code=200
            )
            
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)