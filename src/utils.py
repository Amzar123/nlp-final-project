"""Small helpers shared by every task."""
import json
from pathlib import Path

class Device:
    @staticmethod
    def auto():
        """Best available torch device: cuda, then mps (Apple Silicon), then cpu."""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

class JsonlFile:
    @staticmethod
    def read(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def write(rows, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
