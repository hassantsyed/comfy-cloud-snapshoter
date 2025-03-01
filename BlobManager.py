import logging
import os
import requests  # to call the Next.js API
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ------------------------------------------------------------------------
# These might be settings you customize or pull from environment variables
# ------------------------------------------------------------------------
GET_UPLOAD_URLS_ENDPOINT = "https://comfy-cloud-serverless.vercel.app/api/get-upload-urls"
DEFAULT_CONTAINER_NAME = "custom-nodes-models"

class ModelManager:
    def __init__(self, max_workers: int = 5):
        """
        Initialize the manager. Instead of Azure blob classes,
        we'll rely on your Next.js API for SAS URLs.
        """
        self.max_workers = max_workers

    # ---------------------------------------------------
    #  Example existence check: HEAD request on SAS URL
    # ---------------------------------------------------
    def _check_remote_existence_via_sas(self, sas_url: str) -> bool:
        """
        Option A: Head request to see if the blob is already there.
        This requires 'read' permissions on the SAS URL.
        If your Next.js route always grants "rw",
        we *could* do a HEAD request on that same URL.
        """
        try:
            resp = requests.head(sas_url, timeout=10)
            if resp.status_code == 200:
                return True
            elif resp.status_code == 404:
                return False
            else:
                logger.warning(f"SAS HEAD check returned {resp.status_code} for {sas_url}")
                return False
        except Exception as e:
            logger.error(f"HEAD request failed on {sas_url}: {e}")
            return False

    # -------------------------------------------------
    #  Request one or more SAS URLs for *upload*
    # -------------------------------------------------
    def _fetch_upload_sas_urls(
        self,
        container_name: str,
        file_names: List[str],
        prefix: str
    ) -> Optional[List[str]]:
        """
        Calls your Next.js API to get SAS URLs for uploading.
        The prefix can be the hash (or any path structure you want).
        """
        try:
            payload = {
                "containerName": container_name,
                "fileNames": file_names,
                "prefix": prefix
            }
            resp = requests.post(GET_UPLOAD_URLS_ENDPOINT, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return data["uploadUrls"]  # an array of SAS strings
        except Exception as e:
            logger.error(f"Failed to fetch upload SAS URLs: {e}")
            return None

    # -------------------------------------------------
    #  Request one or more SAS URLs for *download*
    # -------------------------------------------------
    def _fetch_download_sas_urls(
        self,
        container_name: str,
        file_names: List[str],
        prefix: str
    ) -> Optional[List[str]]:
        """
        You might reuse the same Next.js route for downloads if it sets
        "r" or "rw" permissions. If you prefer separate endpoints or logic,
        modify accordingly.
        """
        try:
            payload = {
                "containerName": container_name,
                "fileNames": file_names,
                "prefix": prefix
            }
            resp = requests.post(GET_UPLOAD_URLS_ENDPOINT, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return data["uploadUrls"]
        except Exception as e:
            logger.error(f"Failed to fetch download SAS URLs: {e}")
            return None

    def store_model(
        self,
        source_path: Path,
        hash_value: str,
        container_name: str = DEFAULT_CONTAINER_NAME,
        prefix: str = None
    ) -> bool:
        """
        Upload the local file at source_path using a SAS URL from your Next.js endpoint.
        
        :param source_path: The file path on disk.
        :param hash_value: By default used as the 'prefix' in Azure. Also used for caching logic.
        :param container_name: Name of the Azure container, default is 'custom-nodes-models'.
        :param prefix: If set, overrides the default of using `hash_value` as the prefix in Azure's path.
        """
        try:
            used_prefix = prefix if prefix else hash_value
            file_name = source_path.name
            sas_urls = self._fetch_upload_sas_urls(
                container_name=container_name,
                file_names=[file_name],
                prefix=used_prefix
            )
            if not sas_urls:
                logger.error("Unable to get SAS URLs for upload.")
                return False

            sas_url = sas_urls[0]

            # Confirm local file is valid
            if not source_path.exists():
                logger.error(f"Source file does not exist: {source_path}")
                return False
            if not os.access(source_path, os.R_OK):
                logger.error(f"Source file is not readable: {source_path}")
                return False

            file_size = source_path.stat().st_size
            logger.info(f"Uploading {source_path} ({file_size/1024/1024:.2f}MB) to {sas_url}")

            # Important: specify 'x-ms-blob-type': 'BlockBlob'
            headers = {
                "x-ms-blob-type": "BlockBlob",
                "Content-Type": "application/octet-stream",
                "Content-Length": str(file_size),
            }

            with open(source_path, "rb") as f:
                put_resp = requests.put(sas_url, data=f, headers=headers, timeout=60)
                put_resp.raise_for_status()

            logger.info(f"Successfully stored file {source_path.name} in container={container_name}, prefix={used_prefix}")
            return True

        except Exception as e:
            logger.error(f"Failed to store file {source_path}: {str(e)}")
            return False

    def restore_model(self, hash_value: str, dest_path: Path) -> bool:
        """
        Download the remote file via a SAS URL and save it at dest_path.
        """
        try:
            file_name = dest_path.name
            sas_urls = self._fetch_download_sas_urls(
                container_name=DEFAULT_CONTAINER_NAME,
                file_names=[file_name],
                prefix=hash_value
            )
            if not sas_urls:
                logger.error("Unable to get SAS URLs for download.")
                return False
            sas_url = sas_urls[0]

            # Check existence by HEAD if desired:
            # if not self._check_remote_existence_via_sas(sas_url):
            #     logger.error(f"Model {hash_value} not found in remote storage (HEAD check).")
            #     return False

            # Create directory if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading {sas_url} to {dest_path}")
            resp = requests.get(sas_url, stream=True, timeout=60)
            resp.raise_for_status()

            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Restored model {hash_value} to {dest_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore model {hash_value}: {str(e)}")
            return False

    def store_models_from_environment(
        self,
        comfy_path: str,
        environment: Dict
    ) -> None:
        """
        Example parallel store, updated to use the new _fetch_upload_sas_urls logic.
        """
        success_count = 0
        error_count = 0
        skipped_count = 0
        total_size = 0

        models = environment.get("custom_nodes_models", [])
        logger.info(f"Found {len(models)} models to store in environment.")
        # List of (Path, hash, size)
        to_upload = []

        for m in models:
            source_path = Path(comfy_path) / "custom_nodes" / m["path"]
            if not source_path.exists():
                logger.error(f"Model file not found: {source_path}")
                error_count += 1
                continue

            # If you want to check caching via HEAD, you'd first get a read SAS and HEAD it
            # (or use a separate route). Simplified approach below:
            # ...
            # skip if it exists

            to_upload.append((source_path, m["hash"], m["size"]))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(self.store_model, p, h): (p, h, sz)
                for (p, h, sz) in to_upload
            }

            for future in as_completed(future_map):
                p, h, sz = future_map[future]
                try:
                    if future.result():
                        success_count += 1
                        total_size += sz
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1
                    logger.error(f"Exception uploading {p}: {e}")

        logger.info(f"Models stored: {success_count} - Skipped: {skipped_count} - Failed: {error_count}")
        logger.info(f"Total size of new models: {total_size / (1024*1024*1024):.2f}GB")

    def restore_models_from_environment(
        self,
        comfy_path: str,
        environment: Dict
    ) -> None:
        """
        Example parallel restore, updated to get SAS URLs for download.
        """
        success_count = 0
        error_count = 0
        skipped_count = 0
        total_size = 0

        models = environment.get("custom_nodes_models", [])
        to_download = []

        for m in models:
            dest_path = Path(comfy_path) / m["path"]
            if dest_path.exists():
                skipped_count += 1
                continue
            to_download.append((m["hash"], dest_path, m["size"]))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(self.restore_model, h, dp): (h, dp, sz)
                for (h, dp, sz) in to_download
            }

            for future in as_completed(future_map):
                h, dp, sz = future_map[future]
                try:
                    if future.result():
                        success_count += 1
                        total_size += sz
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1
                    logger.error(f"Exception restoring {dp}: {e}")

        logger.info(f"Models restored: {success_count} - Skipped: {skipped_count} - Failed: {error_count}")
        logger.info(f"Total size of restored models: {total_size / (1024*1024*1024):.2f}GB")

    def download_environment_file(
        self,
        environment_id: str,
        file_name: str,
        dest_path: Path
    ) -> bool:
        """
        Download a specific file from an environment by ID.

        Args:
            environment_id: UUID of the environment
            file_name: Name of the file to download (e.g., "environment.json" or "comfy_state.zip")
            dest_path: Where to save the downloaded file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Build the download URL directly
            base_url = "https://comfycloud.blob.core.windows.net"
            container_name = "environments"
            blob_path = f"{environment_id}/{file_name}"
            download_url = f"{base_url}/{container_name}/{blob_path}"

            # Log the download attempt
            logger.info(f"Downloading {file_name} from environment {environment_id}")
            logger.debug(f"Download URL: {download_url}")

            # Make the HTTP GET request
            resp = requests.get(download_url, stream=True, timeout=60)

            # Check if file exists
            if resp.status_code == 404:
                logger.error(f"File {file_name} not found for environment {environment_id}")
                return False

            resp.raise_for_status()

            # Create directory if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the file
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Successfully downloaded {file_name} to {dest_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download {file_name} for environment {environment_id}: {str(e)}")
            return False


def store_models(
    comfy_path: str,
    environment: Dict
) -> None:
    """
    Wrapper that creates a ModelManager and stores 'large_files' from environment.
    """
    manager = ModelManager()
    large_files = environment.get("large_files", [])
    if not large_files:
        logger.warning("No large files found in environment.")
        return

    success_count = 0
    error_count = 0
    skipped_count = 0
    total_size = 0
    to_upload = []

    for f in large_files:
        source_path = Path(comfy_path) / f["path"]
        if not source_path.exists():
            logger.error(f"File not found: {source_path}")
            error_count += 1
            continue
        to_upload.append((source_path, f["hash"], f["size"]))

    with ThreadPoolExecutor(max_workers=manager.max_workers) as executor:
        future_map = {
            executor.submit(manager.store_model, p, h): (p, h, sz)
            for (p, h, sz) in to_upload
        }

        for future in as_completed(future_map):
            p, h, sz = future_map[future]
            try:
                if future.result():
                    success_count += 1
                    total_size += sz
                else:
                    error_count += 1
            except Exception as e:
                logger.error(f"Exception storing {p}: {str(e)}")
                error_count += 1

    logger.info(f"Files stored: {success_count}, Skipped: {skipped_count}, Failed: {error_count}")
    logger.info(f"Total size of new files: {total_size/(1024*1024*1024):.2f}GB")


def restore_models(
    comfy_path: str,
    environment: Dict
) -> None:
    """
    Wrapper that creates a ModelManager and restores 'large_files' from environment.
    """
    manager = ModelManager()
    large_files = environment.get("large_files", [])
    if not large_files:
        logger.warning("No large files to restore in environment.")
        return

    success_count = 0
    error_count = 0
    skipped_count = 0
    total_size = 0
    to_download = []

    for f in large_files:
        dest_path = Path(comfy_path) / f["path"]
        if dest_path.exists():
            logger.info(f"File already exists: {dest_path}")
            skipped_count += 1
            continue
        to_download.append((f["hash"], dest_path, f["size"]))

    with ThreadPoolExecutor(max_workers=manager.max_workers) as executor:
        future_map = {
            executor.submit(manager.restore_model, h, dp): (h, dp, sz)
            for (h, dp, sz) in to_download
        }

        for future in as_completed(future_map):
            h, dp, sz = future_map[future]
            try:
                if future.result():
                    success_count += 1
                    total_size += sz
                else:
                    error_count += 1
            except Exception as e:
                logger.error(f"Exception restoring {dp}: {str(e)}")
                error_count += 1

    logger.info(f"Files restored: {success_count}, Skipped: {skipped_count}, Failed: {error_count}")
    logger.info(f"Total size of restored files: {total_size/(1024*1024*1024):.2f}GB")