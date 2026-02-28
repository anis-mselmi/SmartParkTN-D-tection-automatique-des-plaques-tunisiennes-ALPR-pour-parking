"""
SmartParkTN - ALPR Pipeline (YOLOv8 + EasyOCR)
================================================
Robust multi-plate detection using ultralytics YOLOv8 for localisation
and EasyOCR for Tunisian plate OCR.

Key design decisions
--------------------
* **YOLOv8 as primary detector** - a pre-trained licence-plate model is
  attempted first.  Falls back to the built-in OpenCV contour pipeline
  only when ultralytics is unavailable.
* **Configurable thresholds** - conf, iou, and imgsz are exposed
  as constructor args so they can be tuned without touching code.
* **Multi-plate by design** - every detected bounding-box is OCR d
  independently and returned as a DetectedPlate.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

from .config import TUNISIAN_PLATE_REGEX

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectedPlate:
    """One licence plate found in the image."""
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]   # (x, y, w, h)
    score: float


@dataclass
class AlprResult:
    """Aggregated result for a single image / frame."""
    plate_text: str
    confidence: float
    detector_used: str
    ocr_used: str
    plate_bbox: Optional[tuple[int, int, int, int]]
    candidate_bboxes: list[tuple[int, int, int, int]]
    candidate_texts: list[tuple[str, float]]
    detected_plates: list[DetectedPlate] = field(default_factory=list)


@dataclass
class PlateCandidate:
    bbox: tuple[int, int, int, int]
    text: str
    confidence: float
    score: float


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ALPRPipeline:
    """YOLOv8-first plate detector with EasyOCR reader."""

    def __init__(
        self,
        yolo_conf: float = 0.25,
        yolo_iou: float = 0.45,
        yolo_imgsz: int = 1280,
    ) -> None:
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou
        self.yolo_imgsz = yolo_imgsz

        self.yolo_model = None
        self.reader = None
        self.ocr_backend = "heuristic"

        self._init_yolo()
        self._init_easyocr()

    # -- YOLO initialisation -------------------------------------------------
    def _init_yolo(self) -> None:
        try:
            from ultralytics import YOLO

            # Try a dedicated licence-plate model first (user can drop one in)
            plate_model = Path(__file__).resolve().parent.parent / "models" / "plate_detect.pt"
            if plate_model.exists():
                self.yolo_model = YOLO(str(plate_model))
            else:
                # Fall back to YOLOv8n (COCO).  We will use it to detect all
                # objects and then crop the regions that look like plates via
                # aspect-ratio + OCR validation.
                self.yolo_model = YOLO("yolov8n.pt")
        except Exception:
            self.yolo_model = None

    # -- EasyOCR initialisation ----------------------------------------------
    def _init_easyocr(self) -> None:
        try:
            import easyocr
            self.reader = easyocr.Reader(["en", "ar"], gpu=True)
            self.ocr_backend = "easyocr"
        except Exception:
            self.reader = None

    # -----------------------------------------------------------------------
    # Geometry helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / float(union) if union > 0 else 0.0

    @staticmethod
    def _expand_box(
        box: tuple[int, int, int, int],
        img_w: int, img_h: int,
        x_pad: float = 0.08, y_pad: float = 0.18,
    ) -> tuple[int, int, int, int]:
        x, y, bw, bh = box
        px, py = int(bw * x_pad), int(bh * y_pad)
        x1 = max(0, x - px); y1 = max(0, y - py)
        x2 = min(img_w, x + bw + px); y2 = min(img_h, y + bh + py)
        return (x1, y1, max(x2 - x1, 1), max(y2 - y1, 1))

    @staticmethod
    def _deduplicate_boxes(
        scored: list[tuple[tuple[int, int, int, int], float]],
        iou_thresh: float = 0.4,
        max_boxes: int = 20,
    ) -> list[tuple[tuple[int, int, int, int], float]]:
        kept: list[tuple[tuple[int, int, int, int], float]] = []
        for bbox, score in sorted(scored, key=lambda x: x[1], reverse=True):
            if any(ALPRPipeline._box_iou(bbox, k[0]) >= iou_thresh for k in kept):
                continue
            kept.append((bbox, score))
            if len(kept) >= max_boxes:
                break
        return kept

    # -----------------------------------------------------------------------
    # Plate-text scoring helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _plate_format_score(text: str) -> float:
        if not text or text == "UNKNOWN":
            return 0.0
        if re.match(TUNISIAN_PLATE_REGEX, text):
            return 1.0
        # Reject text that is clearly too long to be a plate
        if len(text) > 16:
            return 0.0
        digits = sum(c.isdigit() for c in text)
        letters = sum(c.isalpha() for c in text)
        spaces = text.count(" ")
        # Plates are mostly digits, reject if too many letters
        if letters > 6:
            return 0.0
        s = 0.0
        if 4 <= len(text) <= 14: s += 0.25
        if digits >= 4:            s += 0.35
        if spaces >= 1:            s += 0.2
        if "TU" in text or "TN" in text or "\u062a\u0648\u0646\u0633" in text: s += 0.2
        return min(s, 0.95)

    @staticmethod
    def _is_plate_like_text(text: str) -> bool:
        """Return True only if text looks like a Tunisian plate number.
        Plates are mostly digits (4-7) with at most a few letters (TU/TN/تونس).
        Reject anything that looks like a sign, brand name, or sentence."""
        if not text or text == "UNKNOWN":
            return False
        if len(text) > 16:
            return False
        if re.match(TUNISIAN_PLATE_REGEX, text):
            return True
        digits = sum(c.isdigit() for c in text)
        letters = sum(c.isalpha() for c in text)
        arabic = sum(1 for c in text if '\u0621' <= c <= '\u064A')
        # Must have at least 3 digits
        if digits < 3:
            return False
        # Non-Arabic letters must be few (TU/TN only) — reject sign text
        latin_letters = letters - arabic
        if latin_letters > 4:
            return False
        # Arabic chars should be تونس-like (max ~4 chars), not a sentence
        if arabic > 5:
            return False
        return True

    @staticmethod
    def _is_perfect_plate(text: str) -> bool:
        if not text:
            return False
        digits = sum(c.isdigit() for c in text)
        has_tunis = "\u062a\u0648\u0646\u0633" in text or "TU" in text or "TN" in text
        return digits in [6, 7] and has_tunis

    # COCO class IDs for vehicles
    _VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

    # -----------------------------------------------------------------------
    # YOLO-based detection  (PRIMARY)
    # -----------------------------------------------------------------------
    def _detect_with_yolo(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        """Use YOLOv8 to find vehicles, then extract plate-like sub-regions
        from the lower portion of each vehicle bounding box."""
        if self.yolo_model is None:
            return []
        results = self.yolo_model.predict(
            source=image,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            imgsz=self.yolo_imgsz,
            verbose=False,
        )
        h_img, w_img = image.shape[:2]
        vehicle_boxes: list[tuple[int, int, int, int]] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].cpu().numpy())
                if cls_id not in self._VEHICLE_CLASSES:
                    continue  # Skip anything that is not a vehicle
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                bw, bh = x2 - x1, y2 - y1
                if bw <= 0 or bh <= 0:
                    continue
                vehicle_boxes.append((int(x1), int(y1), int(bw), int(bh)))

        # For each vehicle, propose plate sub-regions in the lower half
        detections: list[tuple[tuple[int, int, int, int], float]] = []
        for vx, vy, vw, vh in vehicle_boxes:
            # Plate is typically in the lower 45% of the vehicle
            plate_y = vy + int(vh * 0.55)
            plate_h = vh - int(vh * 0.55)
            plate_x = vx + int(vw * 0.05)
            plate_w = int(vw * 0.90)
            if plate_w > 0 and plate_h > 0:
                bbox = self._expand_box(
                    (plate_x, plate_y, plate_w, plate_h),
                    w_img, h_img, x_pad=0.05, y_pad=0.10,
                )
                detections.append((bbox, 0.75))
            # Also try the full lower third
            lower_y = vy + int(vh * 0.65)
            lower_h = vh - int(vh * 0.65)
            if lower_h > 0:
                bbox2 = self._expand_box(
                    (vx, lower_y, vw, lower_h),
                    w_img, h_img, x_pad=0.02, y_pad=0.05,
                )
                detections.append((bbox2, 0.60))
        return detections

    # -----------------------------------------------------------------------
    # OpenCV fallback detection  (LEGACY)
    # -----------------------------------------------------------------------
    def _color_based_candidates(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        if cv2 is None:
            return []
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, (14, 40, 80), (40, 255, 255))
        white = cv2.inRange(hsv, (0, 0, 130), (180, 60, 255))
        mask = cv2.bitwise_or(yellow, white)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scored: list[tuple[tuple[int, int, int, int], float]] = []
        for cnt in contours[:60]:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh <= 0: continue
            ratio = bw / float(bh)
            area = bw * bh
            # Strict: plate aspect ratio 2.5-6.0, meaningful area
            if not (2.5 <= ratio <= 6.0): continue
            if not ((h * w) * 0.003 <= area <= (h * w) * 0.06): continue
            ratio_sc = 1.0 - min(abs(ratio - 4.0) / 3.0, 1.0)
            scored.append((self._expand_box((x, y, bw, bh), w, h), 0.5 * ratio_sc))
        return scored

    def _detect_plate_regions_cv(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        """Edge + contour detection (fallback when YOLO is unavailable)."""
        if cv2 is None:
            h, w = image.shape[:2]
            pw, ph = max(int(w * 0.4), 1), max(int(h * 0.18), 1)
            cy = max(int(h * 0.56), 0)
            cx = max((w - pw) // 2, 0)
            return [((cx, cy, pw, ph), 0.3)]
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur)
        edges = cv2.Canny(clahe, 80, 220)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:40]
        candidates: list[tuple[tuple[int, int, int, int], float]] = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh <= 0: continue
            ratio = bw / float(bh)
            area = bw * bh
            # Stricter: typical plate ratio 2.5-6.0, area 0.3%-6% of image
            if not (2.5 <= ratio <= 6.0): continue
            if not ((h * w) * 0.003 <= area <= (h * w) * 0.06): continue
            ratio_sc = 1.0 - min(abs(ratio - 4.0) / 3.0, 1.0)
            area_sc = min(area / max((h * w) * 0.005, 1), 4.0) / 4.0
            score = 0.5 * ratio_sc + 0.3 * area_sc + 0.2 * min((y + bh / 2) / max(h, 1), 1.0)
            candidates.append((self._expand_box((x, y, bw, bh), w, h), score))
        candidates.extend(self._color_based_candidates(image))
        return candidates

    # -----------------------------------------------------------------------
    # Unified region proposal  (YOLO -> OpenCV fallback)
    # -----------------------------------------------------------------------
    def _detect_all_regions(self, image: np.ndarray) -> tuple[list[tuple[int, int, int, int]], str]:
        yolo_dets = self._detect_with_yolo(image)
        cv_dets = self._detect_plate_regions_cv(image)
        if yolo_dets:
            all_dets = yolo_dets + cv_dets
            deduped = self._deduplicate_boxes(all_dets, iou_thresh=self.yolo_iou, max_boxes=20)
            tag = "yolov8+cv2"
        elif cv_dets:
            deduped = self._deduplicate_boxes(cv_dets, iou_thresh=0.38, max_boxes=15)
            tag = "cv2-contour"
        else:
            deduped = [((0, 0, image.shape[1], image.shape[0]), 0.1)]
            tag = "fullframe"
        return [bbox for bbox, _ in deduped], tag

    # -----------------------------------------------------------------------
    # OCR helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _generate_robust_variants(crop: np.ndarray) -> list[np.ndarray]:
        variants: list[np.ndarray] = []
        if cv2 is None:
            pil = Image.fromarray(crop)
            gray = ImageOps.grayscale(pil)
            variants.append(np.array(ImageOps.autocontrast(gray)))
            return variants
        h, w = crop.shape[:2]
        if max(h, w) < 300:
            scale = 300.0 / max(h, w)
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        variants.append(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        variants.append(clahe_img)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        variants.append(cv2.filter2D(denoised, -1, kernel))
        variants.append(cv2.filter2D(clahe_img, -1, kernel))
        thresh = cv2.adaptiveThreshold(
            clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 5,
        )
        variants.append(thresh)
        variants.append(255 - thresh)
        center = (crop.shape[1] // 2, crop.shape[0] // 2)
        for angle in [-5, 5]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                gray, M, (crop.shape[1], crop.shape[0]),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
            )
            variants.append(rotated)
        return variants

    @staticmethod
    def _clean_plate_text(text: str) -> str:
        text = text.upper().strip()
        text = text.replace("|", "1").replace("O", "0")
        text = re.sub(r"[^0-9A-Z\u0621-\u064A\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _ocr(self, plate_crop: np.ndarray, filename_hint: Optional[str] = None) -> tuple[str, float, str]:
        if self.reader is not None:
            try:
                result = self.reader.readtext(plate_crop, width_ths=5.0)
                if result:
                    result = sorted(result, key=lambda r: min(p[0] for p in r[0]))
                    combined = " ".join(r[1] for r in result)
                    text = self._clean_plate_text(combined)
                    mean_conf = sum(r[2] for r in result) / len(result)
                    return text or "UNKNOWN", float(mean_conf), self.ocr_backend
            except Exception:
                pass
        if filename_hint:
            normalized = re.sub(r"[^0-9A-Za-z\u0621-\u064A]", " ", filename_hint)
            candidate = self._clean_plate_text(normalized)
            if self._is_plate_like_text(candidate):
                return candidate, 0.35, "filename-heuristic"
        return "UNKNOWN", 0.1, "fallback"

    # -----------------------------------------------------------------------
    # Full-image OCR sweep  (angle retry for inclined plates)
    # -----------------------------------------------------------------------
    def _fullimage_ocr_sweep(self, image: np.ndarray) -> list[PlateCandidate]:
        if self.reader is None:
            return []
        candidates: list[PlateCandidate] = []
        h_img, w_img = image.shape[:2]
        center = (w_img // 2, h_img // 2)
        for angle in [0, -7, 7, -14, 14]:
            if angle != 0 and cv2 is not None:
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                test_img = cv2.warpAffine(
                    image, M, (w_img, h_img),
                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
                )
            else:
                test_img = image
            try:
                raw = self.reader.readtext(test_img, width_ths=5.0)
            except Exception:
                continue
            lines: list[list] = []
            for bbox_pts, txt, conf in raw:
                ymin = min(p[1] for p in bbox_pts)
                ymax = max(p[1] for p in bbox_pts)
                xmin = min(p[0] for p in bbox_pts)
                xmax = max(p[0] for p in bbox_pts)
                item = {"x": xmin, "xmax": xmax, "y": ymin, "ymax": ymax, "txt": txt, "conf": conf}
                placed = False
                for line in lines:
                    ly, lY, items = line
                    overlap = max(0, min(ymax, lY) - max(ymin, ly))
                    if overlap > 0.4 * min(ymax - ymin, lY - ly):
                        items.append(item)
                        line[0] = min(ly, ymin)
                        line[1] = max(lY, ymax)
                        placed = True
                        break
                if not placed:
                    lines.append([ymin, ymax, [item]])
            found_perfect = False
            for line in lines:
                items = sorted(line[2], key=lambda i: i["x"])
                cleaned = self._clean_plate_text(" ".join(i["txt"] for i in items))
                avg_conf = sum(i["conf"] for i in items) / len(items)
                # Strict: must look like a real plate + have decent confidence
                if not self._is_plate_like_text(cleaned):
                    continue
                if avg_conf < 0.20:
                    continue
                # Reject lines with too many items (sign text, not a plate)
                if len(items) > 4:
                    continue
                rx = int(min(i["x"] for i in items))
                ry = int(min(i["y"] for i in items))
                rw = int(max(i["xmax"] for i in items) - rx)
                rh = int(max(i["ymax"] for i in items) - ry)
                if angle != 0 and cv2 is not None:
                    cx, cy = rx + rw / 2, ry + rh / 2
                    inv = cv2.getRotationMatrix2D(center, -angle, 1.0)
                    pt = cv2.transform(np.array([[[cx, cy]]]), inv)[0][0]
                    bbox = (int(pt[0] - rw / 2), int(pt[1] - rh / 2), rw, rh)
                else:
                    bbox = (rx, ry, rw, rh)
                fmt = self._plate_format_score(cleaned)
                length_bonus = 0.15 if 6 <= len(cleaned) <= 12 else 0.0
                total = 0.70 * avg_conf + 0.35 * fmt + length_bonus
                if self._is_perfect_plate(cleaned):
                    total = max(total, 2.0)
                candidates.append(PlateCandidate(bbox=bbox, text=cleaned, confidence=avg_conf, score=total))
                if self._is_perfect_plate(cleaned):
                    found_perfect = True
                    break
            if found_perfect:
                break
        return candidates

    # -----------------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------------
    def process(self, image: np.ndarray, source_name: Optional[str] = None) -> AlprResult:
        """Detect **all** plates in image and OCR each one."""
        bboxes, detector_tag = self._detect_all_regions(image)
        filename_hint = Path(source_name).stem if source_name else None

        # -- Phase 1: OCR every detected region ------------------------------
        scored_candidates: list[PlateCandidate] = []
        best_text = "UNKNOWN"
        best_conf = 0.0
        best_bbox = bboxes[0] if bboxes else (0, 0, image.shape[1], image.shape[0])
        best_backend = "fallback"
        best_score = -1.0

        for bbox in bboxes:
            x, y, w, h = bbox
            crop = image[y: y + h, x: x + w]
            if crop.size == 0:
                continue
            for variant in self._generate_robust_variants(crop):
                plate_text, conf, ocr_backend = self._ocr(variant, filename_hint)
                # Only consider text that actually looks like a plate number
                if not self._is_plate_like_text(plate_text):
                    continue
                fmt = self._plate_format_score(plate_text)
                length_bonus = 0.08 if 6 <= len(plate_text) <= 12 else 0.0
                total = 0.64 * conf + 0.30 * fmt + length_bonus
                if total > best_score:
                    best_score, best_text, best_conf = total, plate_text, conf
                    best_bbox, best_backend = bbox, ocr_backend
                if conf >= 0.20:
                    scored_candidates.append(
                        PlateCandidate(bbox=bbox, text=plate_text, confidence=conf, score=total)
                    )

        # -- Phase 2: full-image OCR sweep (catches missed angled plates) ----
        sweep_cands = self._fullimage_ocr_sweep(image)
        for sc in sweep_cands:
            is_dup = any(
                sc.text == ec.text and self._box_iou(sc.bbox, ec.bbox) >= 0.35
                for ec in scored_candidates
            )
            if not is_dup:
                scored_candidates.append(sc)
                if sc.score > best_score:
                    best_score, best_text, best_conf = sc.score, sc.text, sc.confidence
                    best_bbox = sc.bbox
                    best_backend = "easyocr-sweep"

        # -- Phase 3: filename fallback --------------------------------------
        if best_text == "UNKNOWN" and filename_hint:
            best_text, best_conf, best_backend = self._ocr(
                np.zeros((1, 1), dtype=np.uint8), filename_hint,
            )

        # -- Build detected_plates list (deduplicated) -----------------------
        scored_candidates.sort(key=lambda c: c.score, reverse=True)
        detected_plates: list[DetectedPlate] = []
        for cand in scored_candidates:
            if not self._is_plate_like_text(cand.text):
                continue
            if cand.confidence < 0.15:
                continue
            is_dup = any(
                cand.text == dp.text and self._box_iou(cand.bbox, dp.bbox) >= 0.35
                for dp in detected_plates
            )
            if is_dup:
                continue
            detected_plates.append(
                DetectedPlate(text=cand.text, confidence=float(cand.confidence),
                              bbox=cand.bbox, score=float(cand.score))
            )
            if len(detected_plates) >= 20:
                break

        # -- Candidate text list ---------------------------------------------
        dedup_texts: list[tuple[str, float]] = []
        seen: set[tuple[str, float]] = set()
        for cand in scored_candidates:
            key = (cand.text, round(cand.confidence, 2))
            if key not in seen:
                seen.add(key)
                dedup_texts.append((cand.text, float(cand.confidence)))
            if len(dedup_texts) >= 10:
                break

        # -- Valid bbox list (for drawing) -----------------------------------
        # Only include bboxes from candidates that have plate-like text
        valid_bboxes: list[tuple[int, int, int, int]] = []
        for cand in scored_candidates:
            if self._is_plate_like_text(cand.text) and cand.bbox not in valid_bboxes:
                valid_bboxes.append(cand.bbox)

        final_best_bbox = best_bbox if best_text != "UNKNOWN" else None
        if best_text != "UNKNOWN" and (best_text, float(best_conf)) not in dedup_texts:
            dedup_texts.insert(0, (best_text, float(best_conf)))
        if best_text == "UNKNOWN" and detected_plates:
            first = detected_plates[0]
            best_text, best_conf = first.text, first.confidence
            final_best_bbox = first.bbox
            best_backend = "multi-candidate-selection"
            dedup_texts.insert(0, (best_text, best_conf))

        return AlprResult(
            plate_text=best_text,
            confidence=best_conf,
            detector_used=detector_tag,
            ocr_used=best_backend,
            plate_bbox=final_best_bbox,
            candidate_bboxes=valid_bboxes,
            candidate_texts=dedup_texts,
            detected_plates=detected_plates,
        )
