"""
Export YOLO model to OpenVINO format for Intel CPU optimization.
Supports FP32 (default) and INT8 (quantized) formats.

Usage:
    python export_openvino.py           # Export FP32
    python export_openvino.py --int8    # Export INT8 (faster, slightly less accurate)
    python export_openvino.py verify    # Verify installation
    python export_openvino.py benchmark # Benchmark all available models
"""

import os
import sys
import logging
import argparse
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
MODEL_PT = os.path.join(MODEL_DIR, "model.pt")
MODEL_OPENVINO_FP32 = os.path.join(MODEL_DIR, "model_openvino_model")
MODEL_OPENVINO_INT8 = os.path.join(MODEL_DIR, "model_int8_openvino_model")
CALIBRATION_DIR = os.path.join(MODEL_DIR, "calibration")
CALIBRATION_YAML = os.path.join(MODEL_DIR, "calibration.yaml")
EXAMPLES_DIR = os.path.join(SCRIPT_DIR, "examples")


def setup_calibration_data():
    """
    Setup calibration data for INT8 quantization.
    Uses example images if calibration folder doesn't exist.
    Returns path to calibration.yaml for ultralytics.
    """
    # Check if calibration YAML exists
    if os.path.exists(CALIBRATION_YAML) and os.path.exists(CALIBRATION_DIR) and os.listdir(CALIBRATION_DIR):
        logger.info(f"Using existing calibration data: {CALIBRATION_YAML}")
        return CALIBRATION_YAML

    # Create calibration folder
    os.makedirs(CALIBRATION_DIR, exist_ok=True)

    # Copy example images to calibration folder
    if os.path.exists(EXAMPLES_DIR):
        copied = 0
        for f in os.listdir(EXAMPLES_DIR):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                src = os.path.join(EXAMPLES_DIR, f)
                dst = os.path.join(CALIBRATION_DIR, f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied += 1

        if copied > 0:
            logger.info(f"Copied {copied} images from examples/ to calibration/")

    # Create calibration.yaml if not exists
    if not os.path.exists(CALIBRATION_YAML):
        with open(CALIBRATION_YAML, 'w') as f:
            f.write(f"""# Calibration dataset for INT8 quantization
path: {CALIBRATION_DIR}
train: .
val: .

names:
  0: bubble
""")
        logger.info(f"Created calibration YAML: {CALIBRATION_YAML}")

    if os.listdir(CALIBRATION_DIR):
        return CALIBRATION_YAML

    logger.warning("No calibration images found. INT8 export may be less optimal.")
    logger.warning(f"Add manga page images to: {CALIBRATION_DIR}")
    return None


def export_openvino_fp32(force: bool = False):
    """
    Export YOLO model to OpenVINO FP32 format.
    """
    from ultralytics import YOLO

    if not os.path.exists(MODEL_PT):
        logger.error(f"Model not found: {MODEL_PT}")
        return None

    if os.path.exists(MODEL_OPENVINO_FP32) and not force:
        logger.info(f"OpenVINO FP32 model already exists: {MODEL_OPENVINO_FP32}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            logger.info("Export cancelled")
            return MODEL_OPENVINO_FP32

    logger.info(f"Loading model: {MODEL_PT}")
    model = YOLO(MODEL_PT)

    logger.info("Exporting to OpenVINO FP32 format...")
    try:
        export_path = model.export(format="openvino", half=False)
        logger.info(f"Successfully exported to: {export_path}")
        return export_path
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return None


def export_openvino_int8(force: bool = False):
    """
    Export YOLO model to OpenVINO INT8 format (quantized).
    Provides ~2x speedup with minimal accuracy loss.
    """
    from ultralytics import YOLO

    if not os.path.exists(MODEL_PT):
        logger.error(f"Model not found: {MODEL_PT}")
        return None

    if os.path.exists(MODEL_OPENVINO_INT8) and not force:
        logger.info(f"OpenVINO INT8 model already exists: {MODEL_OPENVINO_INT8}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            logger.info("Export cancelled")
            return MODEL_OPENVINO_INT8

    # Setup calibration data
    calibration_data = setup_calibration_data()

    logger.info(f"Loading model: {MODEL_PT}")
    model = YOLO(MODEL_PT)

    logger.info("Exporting to OpenVINO INT8 format...")
    logger.info("This may take several minutes...")

    try:
        # Export with INT8 quantization
        export_path = model.export(
            format="openvino",
            int8=True,
            data=calibration_data,  # Use calibration data if available
        )

        # Rename to int8 folder
        if export_path and os.path.exists(export_path):
            if os.path.exists(MODEL_OPENVINO_INT8):
                shutil.rmtree(MODEL_OPENVINO_INT8)
            shutil.move(export_path, MODEL_OPENVINO_INT8)
            logger.info(f"Successfully exported to: {MODEL_OPENVINO_INT8}")
            return MODEL_OPENVINO_INT8

    except Exception as e:
        logger.error(f"INT8 export failed: {e}")
        logger.info("")
        logger.info("Troubleshooting:")
        logger.info("  1. Make sure OpenVINO is installed: pip install openvino")
        logger.info("  2. Add calibration images to: model/calibration/")
        logger.info("  3. Try FP32 export first: python export_openvino.py")
        return None


def verify_openvino():
    """Verify OpenVINO installation and available models."""
    logger.info("=" * 50)
    logger.info("OpenVINO Verification")
    logger.info("=" * 50)

    # Check OpenVINO installation
    try:
        import openvino
        logger.info(f"[OK] OpenVINO version: {openvino.__version__}")
    except ImportError:
        logger.error("[FAIL] OpenVINO not installed")
        logger.info("  Install with: pip install openvino")
        return False

    # Check models
    models_found = []

    if os.path.exists(MODEL_PT):
        size = os.path.getsize(MODEL_PT) / (1024 * 1024)
        logger.info(f"[OK] PyTorch model: {MODEL_PT} ({size:.1f} MB)")
        models_found.append("pytorch")

    if os.path.exists(MODEL_OPENVINO_FP32):
        logger.info(f"[OK] OpenVINO FP32: {MODEL_OPENVINO_FP32}")
        models_found.append("openvino_fp32")
    else:
        logger.warning(f"[--] OpenVINO FP32 not found")

    if os.path.exists(MODEL_OPENVINO_INT8):
        logger.info(f"[OK] OpenVINO INT8: {MODEL_OPENVINO_INT8}")
        models_found.append("openvino_int8")
    else:
        logger.warning(f"[--] OpenVINO INT8 not found")

    # Check calibration data
    if os.path.exists(CALIBRATION_DIR):
        num_images = len([f for f in os.listdir(CALIBRATION_DIR)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        logger.info(f"[OK] Calibration images: {num_images}")
    else:
        logger.warning(f"[--] Calibration folder not found")

    logger.info("=" * 50)
    logger.info(f"Available models: {', '.join(models_found)}")
    logger.info("=" * 50)

    return len(models_found) > 0


def benchmark():
    """Benchmark all available YOLO models."""
    import time
    from ultralytics import YOLO
    from PIL import Image

    test_image = os.path.join(EXAMPLES_DIR, "0.png")
    if not os.path.exists(test_image):
        # Try to find any image
        for f in os.listdir(EXAMPLES_DIR) if os.path.exists(EXAMPLES_DIR) else []:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_image = os.path.join(EXAMPLES_DIR, f)
                break

    if not os.path.exists(test_image):
        logger.error("No test image found in examples/")
        return

    img = Image.open(test_image)
    logger.info(f"Test image: {test_image}")
    logger.info("=" * 60)

    results = {}

    # Benchmark each available model
    models_to_test = [
        ("PyTorch (FP32)", MODEL_PT),
        ("OpenVINO (FP32)", MODEL_OPENVINO_FP32),
        ("OpenVINO (INT8)", MODEL_OPENVINO_INT8),
    ]

    for name, model_path in models_to_test:
        if not os.path.exists(model_path):
            logger.info(f"{name}: Not available")
            continue

        logger.info(f"\nBenchmarking {name}...")
        try:
            model = YOLO(model_path)

            # Warmup
            for _ in range(3):
                model(img, verbose=False)

            # Measure
            times = []
            for _ in range(10):
                start = time.perf_counter()
                model(img, verbose=False)
                times.append(time.perf_counter() - start)

            avg_time = sum(times) / len(times) * 1000
            results[name] = avg_time
            logger.info(f"{name}: {avg_time:.1f}ms")

        except Exception as e:
            logger.error(f"{name}: Failed - {e}")

    # Summary
    if results:
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 60)

        baseline = results.get("PyTorch (FP32)", list(results.values())[0])
        for name, time_ms in sorted(results.items(), key=lambda x: x[1]):
            speedup = baseline / time_ms
            logger.info(f"{name:20} {time_ms:8.1f}ms  ({speedup:.2f}x)")

        logger.info("=" * 60)

        # Recommendation
        if "OpenVINO (INT8)" in results:
            int8_time = results["OpenVINO (INT8)"]
            logger.info(f"\nRecommendation: Use OpenVINO INT8 ({int8_time:.1f}ms)")
        elif "OpenVINO (FP32)" in results:
            fp32_time = results["OpenVINO (FP32)"]
            logger.info(f"\nRecommendation: Use OpenVINO FP32 ({fp32_time:.1f}ms)")
            logger.info("For faster inference, export INT8: python export_openvino.py --int8")


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLO to OpenVINO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python export_openvino.py           # Export FP32 (default)
    python export_openvino.py --int8    # Export INT8 (faster)
    python export_openvino.py --all     # Export both FP32 and INT8
    python export_openvino.py verify    # Verify installation
    python export_openvino.py benchmark # Benchmark all models
        """
    )

    parser.add_argument("command", nargs="?", default="export",
                       choices=["export", "verify", "benchmark"],
                       help="Command to run")
    parser.add_argument("--int8", action="store_true",
                       help="Export INT8 quantized model")
    parser.add_argument("--all", action="store_true",
                       help="Export both FP32 and INT8 models")
    parser.add_argument("--force", "-f", action="store_true",
                       help="Force overwrite existing models")

    args = parser.parse_args()

    if args.command == "verify":
        verify_openvino()
    elif args.command == "benchmark":
        benchmark()
    else:
        # Export
        if args.all:
            logger.info("Exporting FP32 model...")
            export_openvino_fp32(force=args.force)
            logger.info("\nExporting INT8 model...")
            export_openvino_int8(force=args.force)
        elif args.int8:
            export_openvino_int8(force=args.force)
        else:
            export_openvino_fp32(force=args.force)

        logger.info("\nDone! Run 'python export_openvino.py verify' to check status.")
        logger.info("Run 'python export_openvino.py benchmark' to compare performance.")


if __name__ == "__main__":
    main()
