"""Versioned model registry for tracking trained models."""

from datetime import datetime
from pathlib import Path
import json


class ModelRegistry:
    """Keeps track of trained model versions and their metadata."""
    
    def __init__(self, models_dir: str = "./models"):
        self.dir = Path(models_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.dir / "registry.json"
        self.registry = self._load()
    
    def _load(self) -> dict:
        if self.registry_file.exists():
            return json.loads(self.registry_file.read_text())
        return {"models": []}
    
    def _save(self):
        self.registry_file.write_text(json.dumps(self.registry, indent=2))
    
    def register(self, name: str, metrics: dict, path: str):
        """Register a new model version."""
        entry = {
            "name": name,
            "version": len(self.registry["models"]) + 1,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "path": path
        }
        self.registry["models"].append(entry)
        self._save()
    
    def get_latest(self, name: str) -> dict | None:
        """Get the latest version of a named model."""
        matching = [m for m in self.registry["models"] if m["name"] == name]
        return matching[-1] if matching else None
    
    def list_all(self) -> list:
        """List all registered models."""
        return self.registry["models"]
