"""
Model Download Script for Combined Architecture.
Downloads required models for OSB detection, SAM segmentation, etc.

Usage:
    python download_models.py                 # FULL INSTALL (deps + all models)
    python download_models.py status          # Show model status
    python download_models.py realtime        # Download for realtime mode
    python download_models.py quality         # Download for quality mode
    python download_models.py premium         # Download for premium mode
    python download_models.py all             # Download all models
    python download_models.py install         # Install pip dependencies only
"""

import os
import sys
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

# Model definitions with HuggingFace sources (ported from MangaTranslator)
MODELS = {
    "speech_bubble": {
        "description": "Speech Bubble Detector (YOLOv8m-seg)",
        "filename": "yolov8m_seg-speech-bubble.pt",
        "target_dir": "model",
        "huggingface_repo": "kitsumed/yolov8m_seg-speech-bubble",
        "huggingface_file": "model.pt",
        "size_mb": 100,
        "required_for": ["realtime", "quality", "premium"],
    },
    "conjoined_bubble": {
        "description": "Conjoined Bubble Detector (YOLOv8m)",
        "filename": "comic-speech-bubble-detector-yolov8m.pt",
        "target_dir": "model",
        "huggingface_repo": "ogkalu/comic-speech-bubble-detector-yolov8m",
        "huggingface_file": "comic-speech-bubble-detector.pt",
        "size_mb": 50,
        "required_for": ["quality", "premium"],
    },
    "osb_text": {
        "description": "OSB Text Detector (YOLOv12x AnimeText)",
        "filename": "animetext_yolov12x.pt",
        "target_dir": "model",
        "huggingface_repo": "deepghs/AnimeText_yolo",
        "huggingface_file": "yolo12x_animetext/model.pt",
        "size_mb": 200,
        "required_for": ["realtime", "quality", "premium"],
    },
    "panel_detector": {
        "description": "Panel Detector (YOLOv11 Manga109)",
        "filename": "manga109_panel_yolov11.pt",
        "target_dir": "model",
        "huggingface_repo": "deepghs/manga109_yolo",
        "huggingface_file": "v2023.12.07_l_yv11/model.pt",
        "size_mb": 80,
        "required_for": ["quality", "premium"],
    },
    "sam2": {
        "description": "SAM2 (Segment Anything Model 2)",
        "huggingface_repo": "facebook/sam2.1-hiera-large",
        "target_dir": "model/sam",
        "size_mb": 2000,
        "required_for": ["quality", "premium"],
        "install_cmd": "pip install transformers",
    },
    "flux_kontext": {
        "description": "Flux Kontext Inpainting (SDNQ uint4) - OPTIONAL",
        "huggingface_repo": "Disty0/FLUX.1-Kontext-dev-SDNQ-uint4-svd-r32",
        "target_dir": "model/flux",
        "size_mb": 6000,
        "required_for": [],  # Not required by default, optional
        "install_cmd": "pip install diffusers transformers accelerate",
    },
    "flux_klein_4b": {
        "description": "Flux Klein 4B (Text-to-Image) - NOT for inpainting",
        "huggingface_repo": "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
        "target_dir": "model/flux",
        "size_mb": 4000,
        "required_for": [],  # Not required - this is NOT an inpainting model
        "install_cmd": "pip install diffusers transformers accelerate",
    },
    "flux_fill": {
        "description": "FLUX.1-Fill-dev (Official Inpainting Model)",
        "huggingface_repo": "black-forest-labs/FLUX.1-Fill-dev",
        "target_dir": "model/flux",
        "size_mb": 24000,  # ~24GB full model
        "required_for": ["premium"],  # Required for Flux inpainting
        "install_cmd": "pip install diffusers transformers accelerate",
    },
}


def get_base_dir() -> Path:
    """Get base directory of the project."""
    return Path(__file__).parent


def check_model_exists(model_name: str) -> bool:
    """Check if a model is already downloaded."""
    model = MODELS.get(model_name)
    if not model:
        return False

    if model.get("filename"):
        path = get_base_dir() / model["target_dir"] / model["filename"]
        return path.exists()

    if model.get("huggingface_repo"):
        # Check if HuggingFace model is cached
        try:
            from huggingface_hub import try_to_load_from_cache
            result = try_to_load_from_cache(
                model["huggingface_repo"],
                "config.json"
            )
            return result is not None
        except Exception:
            return False

    return False


def download_model(model_name: str, force: bool = False) -> bool:
    """
    Download a specific model.

    Args:
        model_name: Name of the model to download
        force: Force re-download even if exists

    Returns:
        True if successful, False otherwise
    """
    model = MODELS.get(model_name)
    if not model:
        print(f"❌ Unknown model: {model_name}")
        return False

    print(f"\n📦 {model['description']}")
    print(f"   Size: ~{model['size_mb']}MB")

    if not force and check_model_exists(model_name):
        print(f"   ✅ Already downloaded")
        return True

    # Check install command
    if model.get("install_cmd"):
        print(f"   ℹ️  Install dependencies: {model['install_cmd']}")

    # Download from URL
    if model.get("url"):
        return _download_from_url(model)

    # Download from HuggingFace
    if model.get("huggingface_repo"):
        return _download_from_huggingface(model)

    print(f"   ⚠️  No download source available for {model_name}")
    print(f"       Please download manually or check documentation.")
    return False


def _download_from_url(model: dict) -> bool:
    """Download model from direct URL."""
    import urllib.request

    url = model["url"]
    target_dir = get_base_dir() / model["target_dir"]
    target_path = target_dir / model["filename"]

    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"   ⬇️  Downloading from {url[:50]}...")

    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\r   Progress: {percent}%", end="", flush=True)

        urllib.request.urlretrieve(url, target_path, progress_hook)
        print(f"\n   ✅ Downloaded to {target_path}")
        return True

    except Exception as e:
        print(f"\n   ❌ Download failed: {e}")
        return False


def _download_from_huggingface(model: dict) -> bool:
    """Download model from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        repo_id = model["huggingface_repo"]
        target_dir = get_base_dir() / model["target_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"   ⬇️  Downloading from HuggingFace: {repo_id}...")
        print(f"       This may take a while for large models...")

        # Check if specific file is requested
        if model.get("huggingface_file"):
            # Download specific file
            filename = model.get("filename", model["huggingface_file"].split("/")[-1])
            target_path = target_dir / filename

            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=model["huggingface_file"],
                local_dir=str(target_dir),
            )

            # Rename/move if needed
            if downloaded_path != str(target_path):
                import shutil
                downloaded = Path(downloaded_path)
                if downloaded.exists():
                    shutil.move(str(downloaded), str(target_path))

            print(f"   ✅ Downloaded to {target_path}")
        else:
            # Download entire repo
            snapshot_download(
                repo_id,
                local_dir=str(target_dir / repo_id.split("/")[-1]),
                local_dir_use_symlinks=False,
            )
            print(f"   ✅ Downloaded to {target_dir}")

        return True

    except ImportError:
        print(f"   ❌ huggingface_hub not installed")
        print(f"       Run: pip install huggingface_hub")
        return False

    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return False


def download_for_mode(mode: str) -> None:
    """Download all models required for a specific mode."""
    print(f"\n{'='*60}")
    print(f"Downloading models for {mode.upper()} mode")
    print(f"{'='*60}")

    required_models = []
    for name, model in MODELS.items():
        if mode in model.get("required_for", []):
            required_models.append(name)

    if not required_models:
        print(f"No additional models needed for {mode} mode")
        return

    print(f"Required models: {', '.join(required_models)}")

    for model_name in required_models:
        download_model(model_name)


def show_status() -> None:
    """Show status of all models."""
    print("\n" + "="*60)
    print("Model Status")
    print("="*60)

    for name, model in MODELS.items():
        exists = check_model_exists(name)
        status = "✅ Installed" if exists else "❌ Not installed"
        print(f"\n{name}")
        print(f"  Description: {model['description']}")
        print(f"  Size: ~{model['size_mb']}MB")
        print(f"  Status: {status}")
        print(f"  Required for: {', '.join(model['required_for'])}")


def install_dependencies(mode: str = "full") -> bool:
    """Install pip dependencies for specified mode."""
    base_dir = get_base_dir()

    requirements_map = {
        "minimal": "requirements-minimal.txt",
        "full": "requirements.txt",
        "cuda": "requirements-cuda.txt",
    }

    req_file = requirements_map.get(mode, "requirements.txt")
    req_path = base_dir / req_file

    if not req_path.exists():
        print(f"❌ Requirements file not found: {req_path}")
        return False

    print(f"\n{'='*60}")
    print(f"Installing dependencies from {req_file}")
    print(f"{'='*60}")

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_path)],
            check=True
        )
        print(f"\n✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        return False


def detect_gpu() -> Dict[str, Any]:
    """Auto-detect available GPU."""
    result = {
        "cuda_available": False,
        "cuda_name": None,
        "intel_gpu_available": False,
        "intel_gpu_name": None,
    }

    # Check NVIDIA CUDA
    try:
        import torch
        if torch.cuda.is_available():
            result["cuda_available"] = True
            result["cuda_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    # Check Intel GPU via OpenVINO
    try:
        from openvino import Core
        core = Core()
        if "GPU" in core.available_devices:
            result["intel_gpu_available"] = True
            try:
                result["intel_gpu_name"] = core.get_property("GPU", "FULL_DEVICE_NAME")
            except Exception:
                result["intel_gpu_name"] = "Intel GPU"
    except ImportError:
        pass

    return result


def full_install(use_cuda: Optional[bool] = None) -> None:
    """
    Complete installation: dependencies + all models.
    Single command to get everything ready.
    Auto-detects GPU if use_cuda is not specified.
    """
    print("\n" + "="*60)
    print("  MANGA TRANSLATOR - FULL INSTALLATION")
    print("="*60)

    # Auto-detect GPU
    gpu_info = detect_gpu()

    if gpu_info["cuda_available"]:
        print(f"\n  NVIDIA GPU detected: {gpu_info['cuda_name']}")
        if use_cuda is None:
            use_cuda = True
            print("  -> Auto-selecting CUDA installation")
    elif gpu_info["intel_gpu_available"]:
        print(f"\n  Intel GPU detected: {gpu_info['intel_gpu_name']}")
        print("  -> Using OpenVINO optimization")
        use_cuda = False
    else:
        print("\n  No GPU detected, using CPU mode")
        use_cuda = False

    print("\nThis will install:")
    print("  1. Python dependencies" + (" (CUDA)" if use_cuda else ""))
    print("  2. All YOLO detection models")
    print("  3. SAM2 segmentation model")
    print("  4. Flux inpainting models (premium)")
    print("\n  Estimated time: 10-30 minutes (depending on internet)")
    print("  Estimated disk space: ~12GB")
    print("="*60)

    # Step 1: Install pip dependencies
    print("\n\n[STEP 1/2] Installing Python dependencies...")
    print("-"*60)

    dep_mode = "cuda" if use_cuda else "full"
    if not install_dependencies(dep_mode):
        print("\n  Dependency installation had issues, continuing anyway...")

    # Step 2: Download all models
    print("\n\n[STEP 2/2] Downloading all models...")
    print("-"*60)

    success_count = 0
    fail_count = 0

    for model_name in MODELS.keys():
        if download_model(model_name):
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n\n" + "="*60)
    print("INSTALLATION COMPLETE")
    print("="*60)
    print(f"  Models downloaded: {success_count}/{len(MODELS)}")
    if fail_count > 0:
        print(f"  Failed: {fail_count} (run 'python download_models.py status' to check)")

    # Show GPU info
    print("\nHardware:")
    if gpu_info["cuda_available"]:
        print(f"  GPU: {gpu_info['cuda_name']} (CUDA)")
    elif gpu_info["intel_gpu_available"]:
        print(f"  GPU: {gpu_info['intel_gpu_name']} (OpenVINO)")
    else:
        print("  GPU: None (CPU mode)")

    print("\nReady to use! Start the server with:")
    print("   python api_extension.py")
    print("\nOr use Docker:")
    print("   docker-compose up -d")

    return fail_count == 0


def show_help():
    """Show detailed help information."""
    help_text = """
Manga Translator - Model & Dependency Manager
==============================================

QUICK START (Full Installation with Auto GPU Detection):
  python download_models.py              # Auto-detects GPU!

COMMANDS:
  (no args)             Full install (auto-detect GPU)
  status                Show status of all models
  install [--mode X]    Install pip dependencies only
                        Modes: minimal, full (default), cuda
  realtime              Download models for realtime mode (<1s)
  quality               Download models for quality mode (2-3s)
  premium               Download models for premium mode (5-10s)
  all                   Download all models (no dependencies)
  download --model X    Download specific model

OPTIONS:
  --cuda                Force CUDA/GPU installation
  --cpu                 Force CPU-only installation
  --force               Force re-download models

EXAMPLES:
  # Full installation (auto-detects NVIDIA GPU)
  python download_models.py

  # Force CUDA even if not detected
  python download_models.py --cuda

  # Force CPU only (no GPU)
  python download_models.py --cpu

  # Check what's installed
  python download_models.py status

  # Install only realtime mode (minimal)
  python download_models.py install --mode minimal
  python download_models.py realtime

MODES:
  realtime  - Fast translation (<1s), basic detection
  quality   - Better accuracy (2-3s), SAM2 segmentation
  premium   - Best quality (5-10s), AI inpainting

GPU AUTO-DETECTION:
  - NVIDIA CUDA: Automatically detected via PyTorch
  - Intel GPU: Automatically detected via OpenVINO
  - No GPU: Falls back to CPU optimization

For more information, see DEPLOY.md
"""
    print(help_text)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Manga Translator - Full Installation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_models.py              # Full install (recommended)
  python download_models.py --cuda       # Full install with CUDA
  python download_models.py status       # Check installation status
  python download_models.py quality      # Download quality mode models only
        """
    )
    parser.add_argument(
        "command",
        nargs="?",
        default=None,  # Default to full install
        choices=["status", "download", "install", "realtime", "quality", "premium", "all", "help"],
        help="Command to run (default: full install)"
    )
    parser.add_argument(
        "--model",
        help="Specific model to download (for 'download' command)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists"
    )
    parser.add_argument(
        "--mode",
        default="full",
        choices=["minimal", "full", "cuda"],
        help="Installation mode for 'install' command"
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Force CUDA/GPU dependencies (auto-detected if not specified)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only installation (ignore GPU)"
    )

    args = parser.parse_args()

    # Determine CUDA usage
    use_cuda = None  # Auto-detect by default
    if args.cuda:
        use_cuda = True
    elif args.cpu:
        use_cuda = False

    # Default: Full installation when no command given
    if args.command is None:
        full_install(use_cuda=use_cuda)

    elif args.command == "help":
        show_help()

    elif args.command == "status":
        show_status()

    elif args.command == "install":
        mode = "cuda" if args.cuda else args.mode
        install_dependencies(mode)

    elif args.command == "download":
        if args.model:
            download_model(args.model, force=args.force)
        else:
            print("Please specify a model with --model")
            show_status()

    elif args.command in ["realtime", "quality", "premium"]:
        download_for_mode(args.command)

    elif args.command == "all":
        for mode in ["realtime", "quality", "premium"]:
            download_for_mode(mode)

    print("\nDone!")


if __name__ == "__main__":
    main()
