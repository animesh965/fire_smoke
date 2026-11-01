from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    app_name: str = "Industrial Fire & Smoke Monitoring"
    environment: str = "production"
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""
    mongodb_uri: str = ""
    mongodb_database: str = "fire_smoke_industrial"
    camera_mongodb_uri: str = ""
    camera_mongodb_database: str = ""
    model_path: str = "models/fire_smoke_r3_e6_final.pt"
    model_device: str = "0"
    model_confidence: float = 0.45
    model_iou: float = 0.5
    infer_every_n_frames: int = 3
    required_consecutive_hits: int = 3
    event_cooldown_seconds: int = 45
    camera_reconnect_seconds: int = 3
    max_cameras: int = 8
    jpeg_quality: int = 82
    snapshot_dir: str = "data/snapshots"
    alert_email_enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    alert_email_to: str = ""
    smtp_starttls: bool = True
    alert_webhook_url: str = ""
    webhook_timeout_seconds: float = 5.0
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    @property
    def model_file(self) -> Path:
        p = Path(self.model_path)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def snapshots_path(self) -> Path:
        p = Path(self.snapshot_dir)
        return p if p.is_absolute() else ROOT_DIR / p

@lru_cache
def get_settings() -> Settings:
    return Settings()
