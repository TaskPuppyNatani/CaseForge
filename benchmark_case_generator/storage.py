"""Storage management for generator state and output files."""

import json
import hashlib
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

from .models import GeneratedCase, CasePlan


class GeneratorState:
    """Manages the generator state file (private ground truth)."""

    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self._data: dict = {
            "version": "1.0",
            "generated_at": None,
            "model_identifier": None,
            "cases": [],
        }
        self._loaded = False

    def load(self) -> list[GeneratedCase]:
        """Load existing state from file. Returns list of existing cases."""
        if not self.state_file.exists():
            return []
        
        with open(self.state_file, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        
        self._loaded = True
        cases = []
        for case_data in self._data.get("cases", []):
            cases.append(GeneratedCase.from_dict(case_data))
        return cases

    def save(self) -> None:
        """Save current state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def initialize(self, model_identifier: str) -> None:
        """Initialize state with model info."""
        self._data["generated_at"] = datetime.utcnow().isoformat() + "Z"
        self._data["model_identifier"] = model_identifier
        if not self._loaded:
            self._loaded = True

    def add_case(self, case: GeneratedCase) -> None:
        """Add a generated case to the state."""
        self._data["cases"].append(case.to_dict())

    @property
    def case_count(self) -> int:
        return len(self._data.get("cases", []))

    @property
    def model_identifier(self) -> Optional[str]:
        return self._data.get("model_identifier")


class OutputManager:
    """Manages output directory and markdown file writing."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self._next_index: int = 1
        self._existing_files: set[str] = set()

    def initialize(self) -> None:
        """Create output directory and scan for existing files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find existing numbered files
        existing = []
        for f in self.output_dir.glob("*.md"):
            stem = f.stem
            if "_" in stem:
                try:
                    num = int(stem.split("_")[0])
                    existing.append(num)
                except ValueError:
                    pass
        
        if existing:
            self._next_index = max(existing) + 1
            self._existing_files = {f.name for f in self.output_dir.glob("*.md")}

    def get_next_filename(self, name_hint: str) -> str:
        """Generate the next filename."""
        # Sanitize name hint for filesystem
        safe_name = "".join(c for c in name_hint if c.isalnum() or c in " -_").strip()
        safe_name = safe_name.replace(" ", "_").lower()[:30]
        if not safe_name:
            safe_name = "case"
        
        filename = f"{self._next_index:03d}_{safe_name}.md"
        return filename

    def write_case(self, filename: str, content: str) -> str:
        """Write a markdown file and return its SHA-256."""
        filepath = self.output_dir / filename
        
        # Never overwrite
        if filepath.exists():
            raise FileExistsError(f"File {filename} already exists")
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        self._next_index += 1
        self._existing_files.add(filename)
        
        # Compute SHA-256
        sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return sha256

    def file_exists(self, filename: str) -> bool:
        """Check if a file already exists."""
        return filename in self._existing_files or (self.output_dir / filename).exists()
