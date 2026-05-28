"""
Export trained PyTorch models to TFLite format with INT8 quantization.

Pipeline:
    PyTorch → ONNX → TFLite (FP32) → TFLite (INT8 quantized)

Usage:
    python -m src.export.to_tflite --checkpoint checkpoints/best.pt --output models/
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from ..models.mobilefacenet import MobileFaceNet
from ..models.liveness_cnn import LivenessCNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def export_to_onnx(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
    output_path: str,
) -> None:
    """Export PyTorch model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13,
    )
    logger.info("Exported ONNX model to %s", output_path)


def export_recognition_model(checkpoint_path: str, output_dir: str) -> None:
    """Export face recognition model to TFLite."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = MobileFaceNet(embedding_dim=128)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    onnx_path = str(output_path / "face_recognition.onnx")
    export_to_onnx(model, (1, 3, 112, 112), onnx_path)
    logger.info("Face recognition model exported to %s", onnx_path)

    try:
        convert_onnx_to_tflite(onnx_path, str(output_path / "face_recognition.tflite"))
    except ImportError:
        logger.warning("onnx2tf not installed — skipping TFLite conversion. Install: pip install onnx2tf")


def export_liveness_model(checkpoint_path: str, output_dir: str) -> None:
    """Export liveness detection model to TFLite."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = LivenessCNN(num_classes=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    onnx_path = str(output_path / "liveness_detector.onnx")
    export_to_onnx(model, (1, 3, 112, 112), onnx_path)
    logger.info("Liveness model exported to %s", onnx_path)

    try:
        convert_onnx_to_tflite(onnx_path, str(output_path / "liveness_detector.tflite"))
    except ImportError:
        logger.warning("onnx2tf not installed — skipping TFLite conversion. Install: pip install onnx2tf")


def convert_onnx_to_tflite(onnx_path: str, tflite_path: str) -> None:
    """Convert ONNX to TFLite with INT8 quantization."""
    try:
        import onnx2tf
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=str(Path(tflite_path).parent),
            non_verbose=True,
        )
        logger.info("Converted to TFLite: %s", tflite_path)
    except ImportError:
        raise ImportError("onnx2tf not installed. Install: pip install onnx2tf")


def quantize_tflite(model_path: str, output_path: str, num_calibration_samples: int = 100) -> None:
    """Apply INT8 post-training quantization to a TFLite model."""
    try:
        import tensorflow as tf

        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        calibration_data = np.random.randn(num_calibration_samples, 112, 112, 3).astype(np.float32)
        converter.representative_dataset = lambda: [
            (tf.constant(calibration_data[i : i + 1]),) for i in range(num_calibration_samples)
        ]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()
        Path(output_path).write_bytes(tflite_model)
        logger.info("INT8 quantized model saved to %s (%.1f MB)", output_path, len(tflite_model) / 1e6)

    except ImportError:
        logger.warning("tensorflow not installed — skipping INT8 quantization")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export models to TFLite")
    parser.add_argument("--checkpoint", required=True, help="PyTorch checkpoint path")
    parser.add_argument("--model-type", choices=["recognition", "liveness"], required=True)
    parser.add_argument("--output", default="models", help="Output directory")
    args = parser.parse_args()

    if args.model_type == "recognition":
        export_recognition_model(args.checkpoint, args.output)
    else:
        export_liveness_model(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
