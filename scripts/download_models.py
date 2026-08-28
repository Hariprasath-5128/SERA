import os
import time
from huggingface_hub import snapshot_download

def robust_download(repo_id, max_retries=1000, wait_seconds=5):
    print(f"Starting robust download for: {repo_id}")
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}/{max_retries}...")
            snapshot_download(repo_id=repo_id, resume_download=True, local_files_only=False)
            print(f"Successfully downloaded and cached: {repo_id}!")
            return True
        except Exception as e:
            print(f"Network drop detected during download of {repo_id}: {e}")
            print(f"Retrying in {wait_seconds} seconds...\n")
            time.sleep(wait_seconds)
    
    print(f"Failed to download {repo_id} after {max_retries} attempts.")
    return False

if __name__ == "__main__":
    # Force environment config
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.insert(0, project_root)
    
    from dotenv import load_dotenv
    env_path = os.path.join(project_root, ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)

    models_to_download = [
        "BAAI/bge-m3",
        "dmis-lab/biobert-base-cased-v1.1"
    ]
    
    for model in models_to_download:
        success = robust_download(model)
        if not success:
            print(f"ABORTING: Could not complete download for {model}.")
            sys.exit(1)
            
    print("\nALL MODELS SUCCESSFULLY CACHED LOCALLY!")
