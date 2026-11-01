

import argparse
import asyncio
import base64
import csv
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import DESCENDING
from ultralytics import YOLO

from app.config import get_settings

import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from urllib.parse import urlsplit, urlunsplit
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fire_smoke_test")


def to_object_id(value):
    if value is None:
        return None
    try:
        return value if isinstance(value, ObjectId) else ObjectId(str(value))
    except Exception:
        return value


class FireBehaviorValidator:
    """Basic per-track temporal and flame-region heuristic; validate on your footage."""
    def __init__(self, window=12):
        self.history = defaultdict(lambda: deque(maxlen=window))

    def update(self, track_id, frame, box, confidence, t, min_seconds, min_hits):
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if crop.size == 0:
            return False, {"reason": "empty_crop"}

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        warm = (((hue <= 35) | (hue >= 170)) & (sat >= 70) & (val >= 120))
        warm_ratio = float(warm.mean())
        brightness = float(val.mean() / 255.0)
        gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (32, 32))
        hist = self.history[track_id]
        flicker = float(cv2.absdiff(gray, hist[-1]["gray"]).mean() / 255.0) if hist else 0.0
        area = float(max(1, (x2 - x1) * (y2 - y1)))
        aspect = float(max(1, x2 - x1) / max(1, y2 - y1))
        hist.append({
            "t": t, "gray": gray, "area": area, "aspect": aspect,
            "conf": confidence, "warm": warm_ratio, "brightness": brightness,
            "flicker": flicker,
        })

        elapsed = hist[-1]["t"] - hist[0]["t"]
        if len(hist) < min_hits:
            return False, {"hits": len(hist), "elapsed": round(elapsed, 2), "reason": "warming_up"}

        areas = np.array([x["area"] for x in hist])
        aspects = np.array([x["aspect"] for x in hist])
        warms = np.array([x["warm"] for x in hist])
        brights = np.array([x["brightness"] for x in hist])
        flickers = np.array([x["flicker"] for x in hist])
        mean_conf = float(np.mean([x["conf"] for x in hist]))
        area_cv = float(np.std(areas) / max(1.0, np.mean(areas)))
        aspect_cv = float(np.std(aspects) / max(0.01, np.mean(aspects)))
        flicker_mean = float(np.mean(flickers[1:])) if len(flickers) > 1 else 0.0
        signals = sum((
            area_cv >= 0.10,
            aspect_cv >= 0.06,
            float(np.std(warms)) >= 0.015,
            float(np.std(brights)) >= 0.025,
            flicker_mean >= 0.025,
        ))
        confirmed = elapsed >= min_seconds and mean_conf >= 0.45 and signals >= 2
        return confirmed, {
            "hits": len(hist), "elapsed_seconds": round(elapsed, 3),
            "mean_confidence": round(mean_conf, 4), "area_cv": round(area_cv, 4),
            "aspect_cv": round(aspect_cv, 4), "mean_flicker": round(flicker_mean, 4),
            "behaviour_signals": int(signals),
            "reason": "confirmed" if confirmed else "unconfirmed",
        }


class CameraMongo:
    def __init__(self, uri, database):
        self.client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        self.db = self.client[database]

    async def connect(self):
        await self.client.admin.command("ping")
        await self.db.fireevents.create_index([("createdAt", DESCENDING)])
        await self.db.fireevents.create_index([("cameraId", 1), ("createdAt", DESCENDING)])

    async def get_camera(self, code=None, camera_id=None):
        query = {"code": code} if code else {"_id": to_object_id(camera_id)}
        camera = await self.db.cameras.find_one(query)
        if not camera:
            raise RuntimeError(f"Camera not found: {query}")
        if not camera.get("isActive", True):
            raise RuntimeError("Camera is marked inactive")
        if not camera.get("rtspUrl"):
            raise RuntimeError("Camera document has no rtspUrl")
        return camera

    async def insert_event(self, camera, image_uri, confidence, track_id, metrics, snapshot_path):
        now = datetime.now(timezone.utc)
        doc = {
            "cameraId": to_object_id(camera.get("_id")),
            "userId": to_object_id(camera.get("userId")),
            "dvrId": to_object_id(camera.get("dvrId")),
            "eventType": "fire detection",
            "image": image_uri,
            "isRead": False,
            "confidence": float(confidence),
            "createdAt": now,
            "updatedAt": now,
            "cameraName": camera.get("name"),
            "cameraCode": camera.get("cameraCode") or camera.get("code"),
            "cameraLocation": camera.get("cameraLocation") or camera.get("location"),
            "trackId": track_id,
            "snapshotPath": str(snapshot_path),
            "behaviorMetrics": metrics,
        }
        result = await self.db.fireevents.insert_one(doc)
        return str(result.inserted_id)

    def close(self):
        self.client.close()


async def run(args):
    cfg = get_settings()
    store = None
    camera = None

    if args.camera_code or args.camera_id:
        # Separate from app.database.py's normal application database connection.
        # Configure CAMERA_MONGODB_URI in .env; never commit a real server address.
        uri = args.mongo_uri or cfg.camera_mongodb_uri
        db_name = args.mongo_database or cfg.camera_mongodb_database
        store = CameraMongo(uri, db_name)
        await store.connect()
        camera = await store.get_camera(args.camera_code, args.camera_id)
        source = camera["rtspUrl"]
        print(f"Camera: {camera.get('name')} ({camera.get('code')}), _id={camera.get('_id')}")
        if camera.get("isOnline") is False:
            print("Warning: database says camera is offline; testing RTSP connection anyway.")
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        if isinstance(source, str) and not Path(source).is_file():
            raise FileNotFoundError(f"Video not found: {source}")

    model_path = Path(args.model)
    if not model_path.is_file():
        if store:
            store.close()
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = YOLO(str(model_path))
    print("Model loaded:", model_path, "| classes:", model.names)

    out = Path(args.output)
    shots = out / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG if isinstance(source, str) and source.startswith("rtsp://") else cv2.CAP_ANY)
    if not cap.isOpened():
        if store:
            store.close()
        raise RuntimeError("Could not open source. Check video path, RTSP reachability, and credentials.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        if store:
            store.close()
        raise RuntimeError("Could not determine frame dimensions.")
    video_path = out / "annotated_video.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        if store:
            store.close()
        raise RuntimeError(f"Cannot create {video_path}")

    local_camera = camera or {
        "_id": None, "name": "Local test source", "code": "LOCAL_TEST",
        "location": None, "userId": None, "dvrId": None,
    }
    validator = FireBehaviorValidator()
    last_detections = []
    saved_events, saved_shots = 0, 0
    last_event_by_track = {}
    frame_no, infer_calls = 0, 0
    started = time.monotonic()
    csv_path = out / "detections.csv"

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            cw = csv.writer(f)
            cw.writerow(["frame", "time_seconds", "class", "confidence", "x1", "y1", "x2", "y2", "track_id", "confirmed"])
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_no += 1
                video_time = (frame_no - 1) / fps

                # Read every frame, but run YOLO only every Nth frame.
                if frame_no == 1 or frame_no % args.infer_every == 0:
                    result = model.track(
                        source=frame, persist=True, tracker="bytetrack.yaml",
                        imgsz=args.imgsz, conf=args.conf, iou=cfg.model_iou, verbose=False,
                    )[0]
                    infer_calls += 1
                    last_detections = []
                    if result.boxes is not None:
                        for b in result.boxes:
                            class_id = int(b.cls[0].item())
                            label = str(model.names[class_id])
                            normalized = label.lower().replace("_", " ").replace("-", " ")
                            if "fire" not in normalized and "smoke" not in normalized:
                                continue
                            conf = float(b.conf[0].item())
                            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                            tid = int(b.id[0].item()) if b.id is not None else None
                            confirmed, metrics = (False, {"reason": "untracked"})
                            if "fire" in normalized and tid is not None:
                                confirmed, metrics = validator.update(
                                    tid, frame, (x1, y1, x2, y2), conf,
                                    video_time, args.confirm_seconds, args.confirm_hits,
                                )
                            det = {
                                "label": label, "normalized": normalized, "conf": conf,
                                "box": (x1, y1, x2, y2), "tid": tid,
                                "confirmed": confirmed, "metrics": metrics,
                            }
                            last_detections.append(det)
                            cw.writerow([frame_no, round(video_time, 3), label, round(conf, 4),
                                         x1, y1, x2, y2, tid, confirmed])

                for d in last_detections:
                    x1, y1, x2, y2 = d["box"]
                    color = (0, 0, 255) if d["confirmed"] else ((0, 165, 255) if "fire" in d["normalized"] else (255, 140, 0))
                    tag = f"{d['label'].upper()} ID:{d['tid']} {d['conf']:.0%}"
                    tag += " CONFIRMED" if d["confirmed"] else (" CANDIDATE" if "fire" in d["normalized"] else "")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, tag, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                    if d["confirmed"] and d["tid"] is not None:
                        now = time.monotonic()
                        if now - last_event_by_track.get(d["tid"], -1e9) >= args.event_cooldown:
                            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            snap = shots / f"fire_{stamp}_track{d['tid']}.jpg"
                            if cv2.imwrite(str(snap), frame):
                                saved_shots += 1
                                last_event_by_track[d["tid"]] = now
                                if store:
                                    encoded = base64.b64encode(snap.read_bytes()).decode("ascii")
                                    data_uri = "data:image/jpeg;base64," + encoded
                                    try:
                                        event_id = await store.insert_event(
                                            local_camera, data_uri, d["conf"], d["tid"], d["metrics"], snap
                                        )
                                        saved_events += 1
                                        print(f"[MONGODB EVENT] _id={event_id}; screenshot={snap}")
                                    except Exception:
                                        log.exception("Failed to insert fire event")
                                else:
                                    print(f"[VIDEO TEST EVENT] screenshot={snap}")

                writer.write(frame)
                if not args.no_display:
                    cv2.imshow("Fire & Smoke Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if frame_no % 100 == 0:
                    print(f"Frames={frame_no}; inference_calls={infer_calls}")
    finally:
        cap.release()
        writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        if store:
            store.close()

    elapsed = time.monotonic() - started
    print("\n===== SUMMARY =====")
    print(f"Frames read: {frame_no}; inference calls: {infer_calls}")
    print(f"Screenshots saved: {saved_shots}; MongoDB events inserted: {saved_events}")
    print(f"Video: {video_path}\nCSV: {csv_path}\nScreenshots: {shots}")
    print(f"Elapsed: {elapsed:.2f}s; read FPS: {frame_no / elapsed:.2f}" if elapsed else "")


def main():
    cfg = get_settings()
    p = argparse.ArgumentParser(description="Test YOLO fire/smoke detection on video, webcam, or MongoDB camera.")
    p.add_argument("--model", default=cfg.model_path)
    p.add_argument("--source", default="videos/f_candle.mp4", help="Video path or webcam index")
    p.add_argument("--camera-code", help="Camera code in servilience_db.cameras, e.g. CAM_01")
    p.add_argument("--camera-id", help="Camera ObjectId in servilience_db.cameras")
    p.add_argument("--mongo-uri", help="Surveillance MongoDB URI; defaults to CAMERA_MONGODB_URI")
    p.add_argument("--mongo-database", default=None)
    p.add_argument("--conf", type=float, default=cfg.model_confidence)
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--infer-every", type=int, default=cfg.infer_every_n_frames)
    p.add_argument("--confirm-seconds", type=float, default=1.5)
    p.add_argument("--confirm-hits", type=int, default=cfg.required_consecutive_hits)
    p.add_argument("--event-cooldown", type=float, default=cfg.event_cooldown_seconds)
    p.add_argument("--output", default="fire_smoke_test_results")
    p.add_argument("--no-display", action="store_true")
    a = p.parse_args()
    if not 0 <= a.conf <= 1:
        p.error("--conf must be between 0 and 1")
    if a.infer_every < 1:
        p.error("--infer-every must be >= 1")
    if a.confirm_hits < 2 or a.confirm_seconds < 0 or a.event_cooldown < 0:
        p.error("confirm-hits must be >= 2; durations must be non-negative")
    if a.camera_code and a.camera_id:
        p.error("Use only one of --camera-code and --camera-id")
    asyncio.run(run(a))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# FastAPI testing API. The original CLI code above is retained.
# ---------------------------------------------------------------------------

API_BASE_DIR = Path(__file__).resolve().parent
_configured_output_dir = Path(os.getenv("FIRE_OUTPUT_DIR", "fire_smoke_test_results"))
API_OUTPUT_DIR = _configured_output_dir if _configured_output_dir.is_absolute() else API_BASE_DIR / _configured_output_dir
API_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_api_model = None
_api_mongo_client = None
_api_db = None
_api_cfg = None


def _configured_confidence():
    """Use .env first, then the existing application setting, then a safe default."""
    raw = os.getenv("FIRE_MODEL_CONFIDENCE")
    if raw not in (None, ""):
        try:
            value = float(raw)
            if 0.0 <= value <= 1.0:
                return value
            log.warning("FIRE_MODEL_CONFIDENCE must be in [0, 1]; using application default")
        except ValueError:
            log.warning("Invalid FIRE_MODEL_CONFIDENCE=%r; using application default", raw)
    cfg = _api_cfg
    try:
        value = float(getattr(cfg, "model_confidence", 0.58))
        return value if 0.0 <= value <= 1.0 else 0.58
    except (TypeError, ValueError):
        return 0.58


def _configured_image_size():
    raw = os.getenv("FIRE_MODEL_IMAGE_SIZE")
    if raw not in (None, ""):
        try:
            value = int(raw)
            if 320 <= value <= 1536:
                return value
            log.warning("FIRE_MODEL_IMAGE_SIZE must be between 320 and 1536; using 768")
        except ValueError:
            log.warning("Invalid FIRE_MODEL_IMAGE_SIZE=%r; using 768", raw)
    return 768


def _configured_int(name, fallback, minimum=1, maximum=None):
    raw = os.getenv(name)
    try:
        value = int(raw) if raw not in (None, "") else int(fallback)
        if value < minimum or (maximum is not None and value > maximum):
            return int(fallback)
        return value
    except (TypeError, ValueError):
        return int(fallback)


def _configured_float(name, fallback, minimum=0.0):
    raw = os.getenv(name)
    try:
        value = float(raw) if raw not in (None, "") else float(fallback)
        return value if value >= minimum else float(fallback)
    except (TypeError, ValueError):
        return float(fallback)


def _apply_rtsp_lan_override(source):
    """Optionally replace the RTSP host/port with the configured LAN DVR address."""
    if not source or os.getenv("FIRE_RTSP_USE_LAN_HOST", "false").strip().lower() not in {"1", "true", "yes", "on"}:
        return source
    try:
        parsed = urlsplit(str(source))
        if parsed.scheme.lower() not in {"rtsp", "rtsps"}:
            return source
        host = os.getenv("FIRE_RTSP_LAN_HOST", "").strip()
        port = os.getenv("FIRE_RTSP_LAN_PORT", "").strip()
        if not host:
            return source
        credentials = ""
        if parsed.username is not None:
            credentials = parsed.username
            if parsed.password is not None:
                credentials += ":" + parsed.password
            credentials += "@"
        hostname = f"[{host}]" if ":" in host and not host.startswith("[") else host
        netloc = credentials + hostname
        if port:
            netloc += ":" + port
        elif parsed.port:
            netloc += ":" + str(parsed.port)
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    except Exception:
        log.exception("Could not apply RTSP LAN host override")
        return source


def _api_mongo_settings():
    cfg = get_settings()
    uri = (
        __import__("os").getenv("FIRE_CAMERA_MONGODB_URI")
        or getattr(cfg, "camera_mongodb_uri", "")
        or __import__("os").getenv("MONGODB_URI", "")
    )
    db_name = (
        __import__("os").getenv("FIRE_CAMERA_MONGODB_DATABASE")
        or getattr(cfg, "camera_mongodb_database", "")
        or "servilience_db"
    )
    if not uri:
        raise RuntimeError(
            "MongoDB URI missing. Set FIRE_CAMERA_MONGODB_URI in .env."
        )
    return cfg, uri, db_name


@asynccontextmanager
async def api_lifespan(app: FastAPI):
    global _api_model, _api_mongo_client, _api_db, _api_cfg

    _api_cfg, mongo_uri, mongo_db_name = _api_mongo_settings()
    model_path = Path(
        __import__("os").getenv("FIRE_MODEL_PATH", _api_cfg.model_path)
    )
    if not model_path.is_absolute():
        model_path = API_BASE_DIR / model_path
    if not model_path.is_file():
        raise RuntimeError(f"YOLO model not found: {model_path}")

    log.info("Loading fire detection model: %s", model_path)
    _api_model = await asyncio.to_thread(YOLO, str(model_path))
    log.info("Fire detection model loaded. Classes: %s", _api_model.names)

    _api_mongo_client = AsyncIOMotorClient(
        mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000
    )
    _api_db = _api_mongo_client[mongo_db_name]
    log.warning("========== FIRE API DATABASE DEBUG ==========")
    log.warning("MongoDB database configured as: %s", mongo_db_name)
    log.warning("MongoDB database actually selected: %s", _api_db.name)
    log.warning("Cameras collection database: %s", _api_db.name)
    log.warning("=============================================")
    try:
        await _api_mongo_client.admin.command("ping")
        await _api_db.fireevents.create_index([("createdAt", DESCENDING)])
        await _api_db.fireevents.create_index([("cameraId", 1), ("createdAt", DESCENDING)])
        log.info("Connected to MongoDB database %s", mongo_db_name)
        yield
    finally:
        if _api_mongo_client is not None:
            _api_mongo_client.close()
        _api_model = None
        _api_mongo_client = None
        _api_db = None


app = FastAPI(
    title="Fire & Smoke Detection Testing API",
    version="1.0.0",
    description="Test fire/smoke detection using image uploads, video uploads, or live RTSP cameras.",
    lifespan=api_lifespan,
)
app.mount("/results", StaticFiles(directory=str(API_OUTPUT_DIR)), name="results")


class LiveCameraRequest(BaseModel):
    rtsp_url: Optional[str] = Field(default=None, description="RTSP URL; use this or camera_id/camera_code")
    camera_id: Optional[str] = None
    camera_code: Optional[str] = None
    duration_seconds: int = Field(default=15, ge=1, le=300)
    infer_every: Optional[int] = Field(default=None, ge=1, le=100)
    image_size: Optional[int] = Field(default=None, ge=320, le=1536)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    confirm_seconds: Optional[float] = Field(default=None, ge=0)
    confirm_hits: Optional[int] = Field(default=None, ge=2, le=100)
    event_cooldown_seconds: Optional[float] = Field(default=None, ge=0)


def _safe_object_id(value):
    if not value:
        return None
    try:
        return ObjectId(str(value))
    except Exception:
        return None


def _api_detection_payload(det, frame_shape):
    x1, y1, x2, y2 = det["box"]
    return {
        "label": det["label"],
        "confidence": round(float(det["conf"]), 5),
        "class_id": int(det["class_id"]),
        "track_id": det.get("tid"),
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "confirmed": bool(det.get("confirmed", False)),
        "behavior_metrics": det.get("metrics", {}),
    }

async def _save_api_event(
    frame,
    det,
    source_type,
    camera=None,
    frame_number=None,
):
    """Save a confirmed fire event and its screenshot in MongoDB."""
    if _api_db is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB is not connected",
        )

    camera = camera or {}

    # This function is intended for fire events, not smoke-only events.
    if "fire" not in det.get("normalized", ""):
        raise ValueError("Only fire detections can be saved as fire events")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    track_id = det.get("tid")

    snap_path = (
        API_OUTPUT_DIR
        / "screenshots"
        / f"fire_{stamp}_track{track_id if track_id is not None else 'unknown'}.jpg"
    )
    snap_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the annotated frame.
    success = await asyncio.to_thread(
        cv2.imwrite,
        str(snap_path),
        frame,
    )
    if not success:
        raise RuntimeError("Could not write fire-event screenshot")

    # Store the screenshot as a Base64 data URI.
    encoded = await asyncio.to_thread(
        lambda: base64.b64encode(snap_path.read_bytes()).decode("ascii")
    )
    image_uri = "data:image/jpeg;base64," + encoded

    now = datetime.now(timezone.utc)

    camera_id = _safe_object_id(camera.get("_id"))
    user_id = _safe_object_id(camera.get("userId"))
    dvr_id = _safe_object_id(camera.get("dvrId"))

    doc = {
        "cameraId": camera_id,
        "userId": user_id,
        "dvrId": dvr_id,
        "eventType": "fire detection",
        "image": image_uri,
        "isRead": False,
        "confidence": float(det["conf"]),
        "createdAt": now,
        "updatedAt": now,
        "cameraName": camera.get("name") or camera.get("cameraName"),
        "cameraCode": camera.get("code") or camera.get("cameraCode"),
        "cameraLocation": (
            camera.get("location") or camera.get("cameraLocation")
        ),
        "trackId": track_id,
        "snapshotPath": str(snap_path),
        "behaviorMetrics": det.get("metrics", {}),
    }

    result = await _api_db.fireevents.insert_one(doc)

    log.info(
        "Fire event saved: event_id=%s camera=%s track_id=%s",
        result.inserted_id,
        doc["cameraName"],
        track_id,
    )

    return str(result.inserted_id), str(snap_path)


def _draw_detections(frame, detections):
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = (0, 0, 255) if det.get("confirmed") else (0, 165, 255)
        tag = f"{det['label'].upper()} {det['conf']:.0%}"
        if det.get("tid") is not None:
            tag += f" ID:{det['tid']}"
        tag += " CONFIRMED" if det.get("confirmed") else " CANDIDATE"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated, tag, (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    return annotated


async def _infer_frame(frame, confidence, image_size, tracking=False):
    kwargs = {
        "source": frame,
        "imgsz": image_size,
        "conf": confidence,
        "iou": _api_cfg.model_iou,
        "verbose": False,
    }
    if tracking:
        kwargs.update({"persist": True, "tracker": "bytetrack.yaml"})
        results = await asyncio.to_thread(_api_model.track, **kwargs)
    else:
        results = await asyncio.to_thread(_api_model.predict, **kwargs)
    result = results[0]
    detections = []
    if result.boxes is None:
        return detections
    for b in result.boxes:
        class_id = int(b.cls[0].item())
        label = str(_api_model.names[class_id])
        normalized = label.lower().replace("_", " ").replace("-", " ")
        if "fire" not in normalized and "smoke" not in normalized:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        tid = int(b.id[0].item()) if b.id is not None else None
        detections.append({
            "class_id": class_id, "label": label, "normalized": normalized,
            "conf": float(b.conf[0].item()), "box": (x1, y1, x2, y2),
            "tid": tid, "confirmed": False, "metrics": {"reason": "unconfirmed"},
        })
    return detections


@app.get("/health")
async def api_health():
    return {
        "status": "ok" if _api_model is not None and _api_db is not None else "starting",
        "model_loaded": _api_model is not None,
        "mongodb_connected": _api_db is not None,
        "model_classes": _api_model.names if _api_model is not None else None,
    }


@app.get("/api/model")
async def api_model_info():
    if _api_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"loaded": True, "classes": _api_model.names}



@app.post("/api/test/image")
async def test_image(
    file: UploadFile = File(...),
):
    if _api_model is None or _api_db is None:
        raise HTTPException(
            status_code=503,
            detail="Model or MongoDB is not ready"
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Upload an image file"
        )

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty"
        )

    image_array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode uploaded image"
        )

    # Read the existing .env configuration through your settings helpers
    conf = _configured_confidence()
    imgsz = _configured_image_size()

    detections = await _infer_frame(
        frame,
        conf,
        imgsz,
        tracking=False
    )

    annotated = _draw_detections(frame, detections)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = API_OUTPUT_DIR / f"image_{stamp}.jpg"

    success = await asyncio.to_thread(
        cv2.imwrite,
        str(out_path),
        annotated
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to save annotated image"
        )

    saved_events = []

    for det in detections:
        event_id, _ = await _save_api_event(
            annotated,
            det,
            "image_test",
            camera=None,
            frame_number=1
        )
        saved_events.append(event_id)

    return {
        "source_type": "image",
        "filename": file.filename,
        "confidence_threshold": conf,
        "image_size": imgsz,
        "detections_count": len(detections),
        "detections": [
            _api_detection_payload(d, frame.shape)
            for d in detections
        ],
        "annotated_image_url": f"/results/{out_path.name}",
        "mongo_event_ids": saved_events,
        "note": (
            "Single-image detections are saved as unconfirmed candidates; "
            "temporal confirmation requires video/live input."
        ),
    }


@app.post("/api/test/video")
async def test_video(
    file: UploadFile = File(...),
):
    if _api_model is None or _api_db is None:
        raise HTTPException(
            status_code=503,
            detail="Model or MongoDB is not ready",
        )

    suffix = Path(file.filename or "upload.mp4").suffix.lower()

    if suffix not in {
        ".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"
    }:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format",
        )

    temp_path = API_OUTPUT_DIR / (
        f"upload_{int(time.time() * 1000)}{suffix}"
    )

    try:
        data = await file.read()

        if not data:
            raise HTTPException(
                status_code=400,
                detail="Uploaded video is empty",
            )

        await asyncio.to_thread(temp_path.write_bytes, data)

        # Read all configuration from .env through existing helpers.
        confidence = _configured_confidence()
        image_size = _configured_image_size()

        infer_every = _configured_int(
            "FIRE_INFER_EVERY", 3, 1, 100
        )
        confirm_seconds = _configured_float(
            "FIRE_CONFIRM_SECONDS", 1.5
        )
        confirm_hits = _configured_int(
            "FIRE_CONFIRM_HITS", 3, 2, 100
        )
        event_cooldown_seconds = _configured_float(
            "FIRE_EVENT_COOLDOWN_SECONDS", 45
        )

        result = await _process_video_file(
            video_path=temp_path,
            source_type="video_test",
            camera=None,
            confidence=confidence,
            image_size=image_size,
            infer_every=infer_every,
            confirm_seconds=confirm_seconds,
            confirm_hits=confirm_hits,
            event_cooldown_seconds=event_cooldown_seconds,
        )

        return result

    finally:
        temp_path.unlink(missing_ok=True)


async def _process_video_file(
    video_path,
    source_type,
    camera,
    confidence,
    image_size,
    infer_every,
    confirm_seconds,
    confirm_hits,
    event_cooldown_seconds,
):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise HTTPException(
            status_code=400,
            detail="Could not open video/RTSP source",
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise HTTPException(
            status_code=400,
            detail="Could not determine video dimensions",
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    video_out = API_OUTPUT_DIR / f"{source_type}_{stamp}.mp4"

    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise HTTPException(
            status_code=500,
            detail="Could not create annotated video",
        )

    validator = FireBehaviorValidator()
    last_detections = []
    frame_no = 0
    infer_calls = 0
    saved_event_ids = []
    last_event_by_track = {}
    all_detections = []
    started = time.monotonic()

    try:
        while True:
            ok, frame = await asyncio.to_thread(cap.read)

            if not ok:
                break

            frame_no += 1
            video_time = (frame_no - 1) / fps

            # Run inference at the configured interval.
            if frame_no == 1 or frame_no % infer_every == 0:
                last_detections = await _infer_frame(
                    frame,
                    confidence,
                    image_size,
                    tracking=True,
                )
                infer_calls += 1

                for det in last_detections:
                    if (
                        "fire" in det["normalized"]
                        and det.get("tid") is not None
                    ):
                        confirmed, metrics = validator.update(
                            det["tid"],
                            frame,
                            det["box"],
                            det["conf"],
                            video_time,
                            confirm_seconds,
                            confirm_hits,
                        )

                        det["confirmed"] = confirmed
                        det["metrics"] = metrics

                    else:
                        det["metrics"] = {
                            "reason": (
                                "untracked"
                                if det.get("tid") is None
                                else "smoke_candidate"
                            )
                        }

                    all_detections.append(
                        _api_detection_payload(det, frame.shape)
                    )

            # Draw the detections on the current frame.
            annotated = _draw_detections(frame, last_detections)
            writer.write(annotated)

            # Save a screenshot and MongoDB event for confirmed fire.
            for det in last_detections:
                if (
                    det.get("confirmed")
                    and det.get("tid") is not None
                ):
                    now = time.monotonic()
                    tid = det["tid"]

                    if (
                        now - last_event_by_track.get(tid, -1e9)
                        >= event_cooldown_seconds
                    ):
                        event_id, screenshot_path = await _save_api_event(
                            annotated,
                            det,
                            source_type,
                            camera=camera,
                            frame_number=frame_no,
                        )

                        saved_event_ids.append(event_id)
                        last_event_by_track[tid] = now

                        log.info(
                            "Saved confirmed fire event %s; screenshot=%s",
                            event_id,
                            screenshot_path,
                        )

    finally:
        cap.release()
        writer.release()

    elapsed = time.monotonic() - started

    return {
        "source_type": source_type,
        "confidence_threshold": confidence,
        "image_size": image_size,
        "infer_every": infer_every,
        "confirm_seconds": confirm_seconds,
        "confirm_hits": confirm_hits,
        "event_cooldown_seconds": event_cooldown_seconds,
        "frames_processed": frame_no,
        "inference_calls": infer_calls,
        "elapsed_seconds": round(elapsed, 2),
        "detections": all_detections,
        "detections_count": len(all_detections),
        "mongo_event_ids": saved_event_ids,
        "events_saved": len(saved_event_ids),
        "annotated_video_url": f"/results/{video_out.name}",
    }



