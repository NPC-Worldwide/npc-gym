"""
Vision processing for emulator environments.

Provides screen capture, object detection, and scene description
for converting visual game state to LLM-compatible observations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from PIL import Image
import io
import base64


class VisionBackend(Enum):
    """Vision processing backends."""
    SIMPLE = "simple"  # Basic color/region detection
    TEMPLATE = "template"  # OpenCV template matching
    DETR = "detr"  # DETR object detection
    RESNET = "resnet"  # ResNet classification
    VLM = "vlm"  # Vision-language model (LLaVA, etc.)


@dataclass
class VisionConfig:
    """Configuration for vision processing."""
    backend: VisionBackend = VisionBackend.SIMPLE
    model_name: str = ""
    device: str = "cpu"

    # Detection settings
    detection_threshold: float = 0.5
    max_detections: int = 20

    # Template matching
    template_dir: str = ""
    template_scale: float = 1.0

    # VLM settings
    vlm_model: str = "liuhaotian/llava-v1.5-7b"
    vlm_max_tokens: int = 256

    # Caching
    cache_embeddings: bool = True
    cache_size: int = 100


@dataclass
class Detection:
    """A detected object in the screen."""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[int, int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.center is None:
            x, y, w, h = self.bbox
            self.center = (x + w // 2, y + h // 2)


@dataclass
class SceneDescription:
    """High-level description of a game screen."""
    summary: str
    detections: List[Detection] = field(default_factory=list)
    regions: Dict[str, str] = field(default_factory=dict)  # region_name -> description
    raw_text: str = ""  # OCR text if applicable
    embeddings: Optional[Any] = None

    def as_text(self) -> str:
        """Convert to text for LLM."""
        parts = [self.summary]

        if self.detections:
            parts.append("\nDetected objects:")
            for det in self.detections[:10]:
                parts.append(f"  - {det.label} at ({det.center[0]}, {det.center[1]}) [{det.confidence:.2f}]")

        if self.regions:
            parts.append("\nScreen regions:")
            for name, desc in self.regions.items():
                parts.append(f"  - {name}: {desc}")

        if self.raw_text:
            parts.append(f"\nVisible text: {self.raw_text[:200]}")

        return "\n".join(parts)


class VisionProcessor:
    """
    Processes game screens into structured observations.

    Supports multiple backends:
    - Simple: Color-based region analysis
    - Template: OpenCV template matching
    - DETR: Deep object detection
    - VLM: Vision-language model descriptions

    Usage:
        processor = VisionProcessor(VisionConfig(backend=VisionBackend.SIMPLE))
        scene = processor.process(pil_image)
        text_obs = scene.as_text()
    """

    def __init__(self, config: VisionConfig = None):
        self.config = config or VisionConfig()
        self._model = None
        self._processor = None
        self._templates = {}
        self._embedding_cache = {}

    def process(self, image: Image.Image) -> SceneDescription:
        """
        Process a screen image into a scene description.

        Args:
            image: PIL Image of the game screen

        Returns:
            SceneDescription with detections and summary
        """
        if self.config.backend == VisionBackend.SIMPLE:
            return self._process_simple(image)
        elif self.config.backend == VisionBackend.TEMPLATE:
            return self._process_template(image)
        elif self.config.backend == VisionBackend.DETR:
            return self._process_detr(image)
        elif self.config.backend == VisionBackend.RESNET:
            return self._process_resnet(image)
        elif self.config.backend == VisionBackend.VLM:
            return self._process_vlm(image)
        else:
            return self._process_simple(image)

    def _process_simple(self, image: Image.Image) -> SceneDescription:
        """Simple color-based analysis."""
        # Analyze dominant colors and regions
        width, height = image.size

        # Sample regions
        regions = {}
        region_names = ["top-left", "top-right", "bottom-left", "bottom-right", "center"]
        region_coords = [
            (0, 0, width//2, height//2),
            (width//2, 0, width, height//2),
            (0, height//2, width//2, height),
            (width//2, height//2, width, height),
            (width//4, height//4, 3*width//4, 3*height//4),
        ]

        for name, (x1, y1, x2, y2) in zip(region_names, region_coords):
            region = image.crop((x1, y1, x2, y2))
            dominant = self._get_dominant_color(region)
            regions[name] = f"Dominant color: {dominant}"

        # Basic summary
        colors = image.getcolors(maxcolors=256*256*256)
        if colors:
            colors = sorted(colors, key=lambda x: x[0], reverse=True)[:5]
            color_desc = ", ".join([self._color_name(c[1]) for c in colors])
        else:
            color_desc = "varied"

        summary = f"Game screen ({width}x{height}) with colors: {color_desc}"

        return SceneDescription(
            summary=summary,
            regions=regions,
        )

    def _process_template(self, image: Image.Image) -> SceneDescription:
        """Template matching using OpenCV."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            return SceneDescription(summary="OpenCV not installed for template matching")

        # Convert to OpenCV format
        img_array = np.array(image)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        detections = []

        # Load templates if not cached
        if not self._templates and self.config.template_dir:
            self._load_templates()

        # Match each template
        for name, template in self._templates.items():
            result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= self.config.detection_threshold)

            for pt in zip(*locations[::-1]):
                h, w = template.shape
                detections.append(Detection(
                    label=name,
                    confidence=float(result[pt[1], pt[0]]),
                    bbox=(pt[0], pt[1], w, h),
                ))

        summary = f"Found {len(detections)} objects via template matching"
        return SceneDescription(summary=summary, detections=detections)

    def _process_detr(self, image: Image.Image) -> SceneDescription:
        """DETR object detection."""
        try:
            from transformers import DetrImageProcessor, DetrForObjectDetection
            import torch
        except ImportError:
            return SceneDescription(summary="transformers not installed for DETR")

        # Load model if needed
        if self._model is None:
            model_name = self.config.model_name or "facebook/detr-resnet-50"
            self._processor = DetrImageProcessor.from_pretrained(model_name)
            self._model = DetrForObjectDetection.from_pretrained(model_name)
            self._model.to(self.config.device)
            self._model.eval()

        # Process image
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process
        results = self._processor.post_process_object_detection(
            outputs,
            threshold=self.config.detection_threshold
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.cpu().numpy()
            x, y, x2, y2 = box
            detections.append(Detection(
                label=self._model.config.id2label[label.item()],
                confidence=score.item(),
                bbox=(int(x), int(y), int(x2-x), int(y2-y)),
            ))

        summary = f"DETR detected {len(detections)} objects"
        return SceneDescription(summary=summary, detections=detections[:self.config.max_detections])

    def _process_resnet(self, image: Image.Image) -> SceneDescription:
        """ResNet classification."""
        try:
            import torch
            from torchvision import models, transforms
        except ImportError:
            return SceneDescription(summary="torchvision not installed for ResNet")

        # Load model if needed
        if self._model is None:
            self._model = models.resnet18(pretrained=True)
            self._model.to(self.config.device)
            self._model.eval()

            self._processor = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

        # Process
        img_t = self._processor(image)
        batch_t = torch.unsqueeze(img_t, 0).to(self.config.device)

        with torch.no_grad():
            out = self._model(batch_t)

        # Get top predictions
        probs = torch.nn.functional.softmax(out, dim=1)[0]
        _, indices = torch.sort(out, descending=True)
        top_indices = indices[0][:5]

        # ImageNet labels (simplified - would load full mapping)
        labels = [f"class_{idx.item()}" for idx in top_indices]
        confs = [probs[idx].item() for idx in top_indices]

        summary = f"ResNet top prediction: {labels[0]} ({confs[0]:.2%})"
        return SceneDescription(summary=summary)

    def _process_vlm(self, image: Image.Image) -> SceneDescription:
        """Vision-language model description."""
        # This would integrate with LLaVA or similar
        # For now, provide a template
        summary = "VLM processing requires model setup. Image captured for analysis."

        # Convert to base64 for potential API use
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        return SceneDescription(
            summary=summary,
            metadata={"image_b64": img_b64[:100] + "..."},  # Truncated
        )

    def _load_templates(self):
        """Load template images for matching."""
        import os
        try:
            import cv2
        except ImportError:
            return

        if not os.path.isdir(self.config.template_dir):
            return

        for filename in os.listdir(self.config.template_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(self.config.template_dir, filename)
                template = cv2.imread(path, 0)
                if template is not None:
                    name = os.path.splitext(filename)[0]
                    self._templates[name] = template

    def _get_dominant_color(self, image: Image.Image) -> str:
        """Get dominant color from image region."""
        # Resize for speed
        small = image.resize((50, 50))
        colors = small.getcolors(maxcolors=2500)
        if colors:
            most_common = max(colors, key=lambda x: x[0])
            return self._color_name(most_common[1])
        return "unknown"

    def _color_name(self, rgb: Tuple[int, ...]) -> str:
        """Convert RGB to color name."""
        if len(rgb) == 4:
            rgb = rgb[:3]  # Drop alpha
        r, g, b = rgb

        # Simple color naming
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > 200 and g < 100 and b < 100:
            return "red"
        elif r < 100 and g > 200 and b < 100:
            return "green"
        elif r < 100 and g < 100 and b > 200:
            return "blue"
        elif r > 200 and g > 200 and b < 100:
            return "yellow"
        elif r > 200 and g < 100 and b > 200:
            return "magenta"
        elif r < 100 and g > 200 and b > 200:
            return "cyan"
        elif r > 150 and g > 100 and b < 100:
            return "orange"
        elif abs(r - g) < 30 and abs(g - b) < 30:
            return "gray"
        else:
            return f"rgb({r},{g},{b})"

    def describe_for_llm(
        self,
        image: Image.Image,
        prompt: str = "Describe what you see in this game screen."
    ) -> str:
        """
        Generate an LLM-ready description of the screen.

        This is a convenience method that combines detection
        with a formatted text output suitable for LLM prompts.
        """
        scene = self.process(image)
        return scene.as_text()
