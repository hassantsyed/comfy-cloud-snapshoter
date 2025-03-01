import json
from typing import Dict, List, Optional
from pathlib import Path
import logging
import git
import os
import shutil
import zipfile
import hashlib
import subprocess
import sys
import torch
from BlobManager import store_models, restore_models, ModelManager
import uuid
import requests
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read the file in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_git_info(path: Path) -> Optional[Dict[str, str]]:
    """
    Get git repository information for a given path.
    
    Args:
        path: Path to the git repository
        
    Returns:
        Dictionary containing git_url and ref, or None if not a git repository
    """
    try:
        repo = git.Repo(path)
        # Get the remote URL
        git_url = next((url for url in repo.remotes.origin.urls), None)
        # Get the current commit SHA
        git_ref = repo.head.commit.hexsha
        
        return {
            "git_url": git_url,
            "ref": git_ref
        }
    except (git.InvalidGitRepositoryError, git.NoSuchPathError, Exception) as e:
        logger.warning(f"Could not get git info for {path}: {e}")
        return None

def get_pip_dependencies() -> list:
    """Get list of installed pip packages and their versions."""
    try:
        # Run pip list and capture output
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list', '--format=json'],
            capture_output=True,
            text=True,
            check=True
        )
        packages = json.loads(result.stdout)
        return [f"{pkg['name']}=={pkg['version']}" for pkg in packages]
    except Exception as e:
        logger.error(f"Failed to get pip dependencies: {str(e)}")
        return []

def get_model_info(path: Path) -> list:
    """
    Get information about model files in a directory.
    
    Args:
        path: Path to scan for model files
        
    Returns:
        List of model files and their metadata
    """
    model_extensions = {'.pt', '.pth', '.onnx', '.safetensors', '.bin'}
    models = []
    
    # Skip certain directories
    skip_dirs = {'__pycache__', '.git', '.ipynb_checkpoints'}
    
    for root, dirs, files in os.walk(path):
        # Skip unwanted directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if any(file.endswith(ext) for ext in model_extensions):
                full_path = Path(root) / file
                rel_path = full_path.relative_to(path.parent)  # Make path relative to custom_nodes parent
                
                model_info = {
                    'path': str(rel_path),
                    'size': full_path.stat().st_size,
                    'hash': calculate_file_hash(full_path)
                }
                models.append(model_info)
                logger.info(f"Found model: {rel_path} ({model_info['size'] / 1024 / 1024:.1f}MB)")
    
    return models

def get_cuda_version() -> str:
    """Get CUDA version if available."""
    try:
        if torch.cuda.is_available():
            # Get CUDA version from torch
            cuda_version = torch.version.cuda
            return cuda_version
        return "CPU"  # or "N/A" if you prefer
    except Exception as e:
        logger.error(f"Failed to get CUDA version: {str(e)}")
        return "unknown"

def get_pytorch_version() -> str:
    """Get PyTorch version with CUDA suffix if applicable."""
    try:
        version = torch.__version__
        if torch.cuda.is_available():
            # Add CUDA suffix (e.g., "+cu118")
            cuda_version = torch.version.cuda.replace(".", "")
            version = f"{version}+cu{cuda_version}"
        return version
    except Exception as e:
        logger.error(f"Failed to get PyTorch version: {str(e)}")
        return "unknown"

def should_skip_file(file_path: str, size_threshold_mb: float = 10.0) -> bool:
    """
    Check if a file should be skipped during archiving.
    
    Args:
        file_path: Path to the file
        size_threshold_mb: Size threshold in MB for large files
        
    Returns:
        bool: True if the file should be skipped
    """
    # Convert threshold to bytes
    size_threshold = size_threshold_mb * 1024 * 1024
    
    # Skip patterns
    skip_patterns = {
        '__pycache__',
        '.git',
        '.ipynb_checkpoints',
        '.pytest_cache',
        '.DS_Store',
        'thumbs.db',
    }
    
    # Skip by pattern
    if any(pattern in str(file_path) for pattern in skip_patterns):
        return True
    
    # Check file size
    try:
        file_size = os.path.getsize(file_path)
        if file_size > size_threshold:
            return True
    except (OSError, FileNotFoundError):
        # If we can't get the file size, don't skip
        pass
        
    return False

def get_large_files(path: Path, size_threshold_mb: float = 10.0) -> list:
    """
    Get information about large files in a directory.
    
    Args:
        path: Path to scan for large files
        size_threshold_mb: Size threshold in MB
        
    Returns:
        List of large files and their metadata
    """
    # Convert threshold to bytes
    size_threshold = size_threshold_mb * 1024 * 1024
    large_files = []
    
    # Skip certain directories
    skip_dirs = {'__pycache__', '.git', '.ipynb_checkpoints', '.pytest_cache'}
    
    for root, dirs, files in os.walk(path):
        # Skip unwanted directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            full_path = Path(root) / file
            try:
                file_size = full_path.stat().st_size
                if file_size > size_threshold:
                    # Make path relative to ComfyUI root directory
                    comfy_root = path.parent.parent  # Go up from custom_nodes/node_dir to ComfyUI
                    rel_path = full_path.relative_to(comfy_root)
                    
                    file_info = {
                        'path': str(rel_path),
                        'size': file_size,
                        'hash': calculate_file_hash(full_path)
                    }
                    large_files.append(file_info)
                    logger.info(f"Found large file: {rel_path} ({file_info['size'] / 1024 / 1024:.1f}MB)")
            except (OSError, FileNotFoundError):
                # Skip files we can't access
                continue
    
    return large_files

def extract_environment(comfy_path: str, size_threshold_mb: float = 10.0) -> dict:
    """
    Extract environment configuration including large files.
    
    Args:
        comfy_path: Path to the ComfyUI installation
        size_threshold_mb: Size threshold in MB for large files
        
    Returns:
        Dictionary containing environment configuration and large file information
    """
    try:
        custom_nodes_path = Path(comfy_path) / 'custom_nodes'
        
        if not custom_nodes_path.exists():
            raise FileNotFoundError(f"Custom nodes directory not found: {custom_nodes_path}")
        
        # Initialize environment structure
        environment = {
            "comfy": {
                "git_url": "https://github.com/comfyanonymous/ComfyUI.git",
                "ref": "",  # Will be populated if git info available
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "cuda_version": get_cuda_version(),
                "pytorch_version": get_pytorch_version()
            },
            "system_dependencies": [], # this is unimplemented for now
            "pip_dependencies": get_pip_dependencies(),
            "large_files": []
        }
        
        # Process each custom node directory for large files
        for node_dir in custom_nodes_path.iterdir():
            if not node_dir.is_dir() or node_dir.name.startswith('.') or node_dir.name == '__pycache__':
                continue
            
            logger.info(f"Processing node: {node_dir.name}")
            
            # Get large file information for this node
            large_files = get_large_files(node_dir, size_threshold_mb)
            environment["large_files"].extend(large_files)
        
        # Try to get ComfyUI git information
        comfy_git_info = get_git_info(Path(comfy_path))
        if comfy_git_info:
            environment["comfy"].update(comfy_git_info)
        
        return environment
        
    except Exception as e:
        logger.error(f"Error extracting environment: {str(e)}")
        raise

def save_environment_json(
    comfy_path: str,
    output_path: str = "environment.json",
    size_threshold_mb: float = 10.0
) -> None:
    """
    Extract environment configuration and save it to a file.
    
    Args:
        comfy_path: Path to the ComfyUI installation
        output_path: Path where to save the environment.json file
        size_threshold_mb: Size threshold in MB for large files
    """
    try:
        environment = extract_environment(comfy_path, size_threshold_mb)
        
        with open(output_path, 'w') as f:
            json.dump(environment, f, indent=2)
        
        # Log some statistics
        total_files = len(environment["large_files"])
        total_size = sum(file["size"] for file in environment["large_files"])
        
        logger.info(f"Saved environment configuration to {output_path}")
        logger.info(f"Found {total_files} large files (>{size_threshold_mb}MB)")
        logger.info(f"Total large file size: {total_size / 1024 / 1024 / 1024:.1f}GB")
        return environment
        
    except Exception as e:
        logger.error(f"Failed to save environment configuration: {str(e)}")
        raise

def notify_deployment_service(environment_id: str) -> bool:
    """
    Notify the deployment service about a new environment.
    
    Args:
        environment_id: UUID of the environment to deploy
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        url = "https://comfy-cloud-serverless.vercel.app/api/deploy-workflow"
        payload = {"environmentId": environment_id}
        
        logger.info(f"Notifying deployment service for environment ID: {environment_id}")
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            logger.info(f"Deployment service notified successfully: {response.text}")
            return True
        else:
            logger.error(f"Failed to notify deployment service. Status code: {response.status_code}, Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error notifying deployment service: {str(e)}")
        return False

def create_comfy_state_archive(
    comfy_path: str,
    output_path: str = "comfy_state.zip",
    size_threshold_mb: float = 10.0
) -> None:
    """
    Create a ZIP archive of the ComfyUI custom_nodes and input directories, excluding large files.
    
    Args:
        comfy_path: Path to the ComfyUI installation
        output_path: Path where to save the ZIP file
        size_threshold_mb: Size threshold in MB for large files
    """
    try:
        comfy_path = Path(comfy_path)
        
        # Check if directories exist
        custom_nodes_path = comfy_path / 'custom_nodes'
        input_path = comfy_path / 'input'
        
        if not custom_nodes_path.exists():
            logger.warning(f"Custom nodes directory not found: {custom_nodes_path}")
        
        if not input_path.exists():
            logger.warning(f"Input directory not found: {input_path}")
            
        if not custom_nodes_path.exists() and not input_path.exists():
            raise FileNotFoundError(f"Neither custom_nodes nor input directories found in {comfy_path}")
        
        # Count total files for progress tracking
        total_files = 0
        if custom_nodes_path.exists():
            total_files += sum(1 for _ in custom_nodes_path.rglob('*') if _.is_file())
        if input_path.exists():
            total_files += sum(1 for _ in input_path.rglob('*') if _.is_file())
            
        processed_files = 0
        skipped_files = 0
        total_size = 0
        
        logger.info(f"Creating archive from: {comfy_path}")
        logger.info(f"Total files to process: {total_files}")
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Process custom_nodes directory
            if custom_nodes_path.exists():
                for root, _, files in os.walk(custom_nodes_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        
                        # Skip large files and other unwanted files
                        if should_skip_file(file_path, size_threshold_mb):
                            skipped_files += 1
                            continue
                        
                        # Calculate path relative to ComfyUI directory
                        rel_path = os.path.relpath(file_path, comfy_path)
                        
                        # Track file size
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        
                        # Add file to ZIP
                        zipf.write(file_path, rel_path)
                        
                        # Update progress
                        processed_files += 1
                        if processed_files % 100 == 0:  # Log every 100 files
                            logger.info(f"Processed {processed_files} files")
            
            # Process input directory
            if input_path.exists():
                for root, _, files in os.walk(input_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        
                        # Skip large files and other unwanted files
                        if should_skip_file(file_path, size_threshold_mb):
                            skipped_files += 1
                            continue
                        
                        # Calculate path relative to ComfyUI directory
                        rel_path = os.path.relpath(file_path, comfy_path)
                        
                        # Track file size
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        
                        # Add file to ZIP
                        zipf.write(file_path, rel_path)
                        
                        # Update progress
                        processed_files += 1
                        if processed_files % 100 == 0:  # Log every 100 files
                            logger.info(f"Processed {processed_files} files")
        
        logger.info(f"Successfully created archive at: {output_path}")
        logger.info(f"Files archived: {processed_files}")
        logger.info(f"Files skipped: {skipped_files}")
        logger.info(f"Archive size: {os.path.getsize(output_path) / 1024 / 1024:.1f}MB")
        logger.info(f"Total size before compression: {total_size / 1024 / 1024:.1f}MB")
        
    except Exception as e:
        logger.error(f"Failed to create ComfyUI state archive: {str(e)}")
        raise

def restore_comfyui(environment: Dict, comfy_path: str) -> bool:
    """
    Clone and set up ComfyUI based on the git information in environment.json.
    
    Args:
        environment: Environment dictionary containing ComfyUI git info
        comfy_path: Path where to install ComfyUI
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        comfy_info = environment.get("comfy", {})
        git_url = comfy_info.get("git_url", "https://github.com/comfyanonymous/ComfyUI.git")
        git_ref = comfy_info.get("ref")
        
        if not git_ref:
            logger.warning("No git reference found in environment.json, using latest version")
        
        comfy_dir = Path(comfy_path)
        
        # Check if directory already exists
        if comfy_dir.exists():
            logger.info(f"ComfyUI directory already exists at {comfy_path}")
            
            # Check if it's a git repository
            try:
                repo = git.Repo(comfy_path)
                current_url = next((url for url in repo.remotes.origin.urls), None)
                
                # If it's the same repository, just pull and checkout
                if current_url == git_url:
                    logger.info(f"Updating existing ComfyUI repository")
                    repo.remotes.origin.fetch()
                    
                    if git_ref:
                        logger.info(f"Checking out revision {git_ref}")
                        repo.git.checkout(git_ref)
                    
                    return True
                else:
                    logger.warning(f"Existing directory is not the expected ComfyUI repository")
                    return False
            except (git.InvalidGitRepositoryError, git.NoSuchPathError):
                logger.warning(f"Existing directory is not a git repository")
                return False
        
        # Clone the repository
        logger.info(f"Cloning ComfyUI from {git_url}")
        repo = git.Repo.clone_from(git_url, comfy_path)
        
        # Checkout specific revision if provided
        if git_ref:
            logger.info(f"Checking out revision {git_ref}")
            repo.git.checkout(git_ref)
        
        # Install requirements
        requirements_file = Path(comfy_path) / "requirements.txt"
        if requirements_file.exists():
            logger.info("Installing ComfyUI requirements")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                check=True
            )
        
        logger.info(f"ComfyUI successfully set up at {comfy_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to set up ComfyUI: {str(e)}")
        return False

def download_environment_files(environment_id: str, output_dir: Path) -> Dict[str, Path]:
    """
    Download environment.json and comfy_state.zip for a specific environment ID.
    
    Args:
        environment_id: UUID of the environment to download
        output_dir: Directory to save the downloaded files
        
    Returns:
        Dict with paths to the downloaded files
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        manager = ModelManager()
        
        # Prepare file paths
        env_json_path = output_dir / "environment.json"
        state_zip_path = output_dir / "comfy_state.zip"
        
        # Download environment.json
        logger.info(f"Downloading environment.json for ID: {environment_id}")
        env_json_success = manager.download_environment_file(
            environment_id=environment_id,
            file_name="environment.json",
            dest_path=env_json_path
        )
        
        if not env_json_success:
            raise Exception(f"Failed to download environment.json for ID: {environment_id}")
        
        # Download comfy_state.zip
        logger.info(f"Downloading comfy_state.zip for ID: {environment_id}")
        state_zip_success = manager.download_environment_file(
            environment_id=environment_id,
            file_name="comfy_state.zip",
            dest_path=state_zip_path
        )
        
        if not state_zip_success:
            logger.warning(f"Failed to download comfy_state.zip for ID: {environment_id}")
            state_zip_path = None  # Set to None if download failed
        
        return {
            "environment_json": env_json_path,
            "comfy_state_zip": state_zip_path
        }
        
    except Exception as e:
        logger.error(f"Failed to download environment files: {str(e)}")
        raise

# Example usage:
if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="ComfyUI environment snapshot utility")
    parser.add_argument("action", choices=["save", "restore"], help="Action to perform: save or restore")
    parser.add_argument("--comfy-path", default="/content/ComfyUI", help="Path to ComfyUI installation")
    parser.add_argument("--env-file", default="environment.json", help="Path to environment.json file")
    parser.add_argument("--state-file", default="comfy_state.zip", help="Path to comfy_state.zip file")
    parser.add_argument("--size-threshold", type=float, default=10.0, help="Size threshold in MB for large files")
    parser.add_argument("--skip-comfy-install", action="store_true", help="Skip ComfyUI installation during restore")
    parser.add_argument("--environment-id", help="Environment ID to restore (UUID format)")
    parser.add_argument("--notify-deployment", action="store_true", help="Notify deployment service after saving")
    
    args = parser.parse_args()
    comfy_path = args.comfy_path
    
    try:
        if args.action == "save":
            # Save environment and large files
            logger.info(f"Saving environment from {comfy_path}")
            environment_state = save_environment_json(
                comfy_path=comfy_path,
                output_path=args.env_file,
                size_threshold_mb=args.size_threshold
            )
            store_models(comfy_path, environment_state)
            create_comfy_state_archive(
                comfy_path=comfy_path,
                output_path=args.state_file,
                size_threshold_mb=args.size_threshold
            )
            
            # Upload environment.json and comfy_state.zip to blob storage
            manager = ModelManager()
            upload_uuid = str(uuid.uuid4())
            logger.info(f"Uploading environment artifacts under prefix {upload_uuid}")

            # Upload environment.json
            env_json_path = Path(args.env_file)
            if env_json_path.exists():
                success = manager.store_model(
                    env_json_path,               # local path
                    hash_value="envjson",        # fallback if prefix not given
                    container_name="environments",
                    prefix=upload_uuid    # environments/uuid folder in Azure
                )
                if success:
                    logger.info(f"Uploaded environment.json to environments/{upload_uuid}/environment.json")
            
            # Upload comfy_state.zip
            zip_path = Path(args.state_file)
            if zip_path.exists():
                success = manager.store_model(
                    zip_path,
                    hash_value="comfyzip",
                    container_name="environments",
                    prefix=upload_uuid
                )
                if success:
                    logger.info(f"Uploaded comfy_state.zip to environments/{upload_uuid}/comfy_state.zip")

            logger.info(f"Environment snapshot saved with ID: {upload_uuid}")
            
            # Notify deployment service if requested
            if args.notify_deployment:
                if notify_deployment_service(upload_uuid):
                    logger.info("Deployment service notified successfully")
                else:
                    logger.warning("Failed to notify deployment service")
            
        elif args.action == "restore":
            # Restore environment and large files
            logger.info(f"Restoring environment to {comfy_path}")
            
            # If environment ID is provided, download the files first
            if args.environment_id:
                logger.info(f"Restoring from environment ID: {args.environment_id}")
                temp_dir = Path("./temp_env_files")
                downloaded_files = download_environment_files(args.environment_id, temp_dir)
                
                # Update file paths to use the downloaded files
                args.env_file = str(downloaded_files["environment_json"])
                if downloaded_files["comfy_state_zip"]:
                    args.state_file = str(downloaded_files["comfy_state_zip"])
                else:
                    logger.warning("comfy_state.zip was not downloaded, will skip state restoration")
            
            # Load environment.json
            try:
                with open(args.env_file, 'r') as f:
                    environment_state = json.load(f)
            except FileNotFoundError:
                logger.error(f"Environment file not found: {args.env_file}")
                sys.exit(1)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in environment file: {e}")
                sys.exit(1)
            
            # 1. Install ComfyUI if needed
            if not args.skip_comfy_install:
                logger.info("Step 1: Setting up ComfyUI")
                if not restore_comfyui(environment_state, comfy_path):
                    logger.error("Failed to set up ComfyUI, aborting restore")
                    sys.exit(1)
            else:
                logger.info("Step 1: Skipping ComfyUI installation (--skip-comfy-install flag used)")
            
            # 2. Extract comfy state archive if it exists
            logger.info("Step 2: Extracting ComfyUI state (custom_nodes and input directories)")
            if os.path.exists(args.state_file):
                logger.info(f"Extracting ComfyUI state from {args.state_file}")
                with zipfile.ZipFile(args.state_file, 'r') as zipf:
                    zipf.extractall(Path(comfy_path))
                logger.info("ComfyUI state extracted successfully")
            else:
                logger.warning(f"ComfyUI state archive not found: {args.state_file}")
            
            # 3. Restore large files from Azure
            logger.info("Step 3: Restoring large files from Azure Blob Storage")
            restore_models(comfy_path, environment_state)
            
            # 4. Install custom node dependencies using ComfyUI-Manager
            logger.info("Step 4: Installing custom node dependencies")
            cm_cli_path = Path(comfy_path) / "custom_nodes" / "ComfyUI-Manager" / "cm-cli.py"
            if cm_cli_path.exists():
                try:
                    # Install ComfyUI Manager requirements before running cm-cli restore
                    logger.info("Installing ComfyUI-Manager requirements")
                    requirements_file = cm_cli_path.parent / "requirements.txt"
                    if requirements_file.exists():
                        subprocess.run(
                            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                            check=True
                        )
                    else:
                        logger.warning(f"ComfyUI-Manager requirements.txt not found at {requirements_file}")
                    
                    logger.info("Running ComfyUI-Manager dependency restoration")
                    subprocess.run(
                        [sys.executable, str(cm_cli_path), "restore-dependencies"],
                        check=True,
                        cwd=comfy_path  # Set working directory to ComfyUI root
                    )
                    logger.info("Custom node dependencies installed successfully")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install custom node dependencies: {str(e)}")
            else:
                logger.warning("ComfyUI-Manager not found, skipping dependency installation")
            
            # Clean up temporary files if we downloaded them
            if args.environment_id and temp_dir.exists():
                logger.info("Cleaning up temporary files")
                shutil.rmtree(temp_dir)
            
            logger.info("Restore operation completed successfully")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1) 