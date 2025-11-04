"""Parse kata manifest.toml files into structured data.

This is mainly for validation. Rust will parse manifests directly,
but we provide this for Python-side tooling if needed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import toml

type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]


@dataclass
class KataVariation:
    """Represents a kata variation with different parameters."""

    name: str
    description: str
    params: dict[str, JSONValue]


@dataclass
class KataManifest:
    """Represents a parsed kata manifest with metadata and variations."""

    name: str
    category: str
    base_difficulty: int
    description: str
    dependencies: list[str]
    variations: list[KataVariation]

    @classmethod
    def from_file(cls, path: Path) -> KataManifest:
        """Load and validate manifest.toml.

        Args:
            path: path to manifest.toml file

        Returns:
            parsed and validated KataManifest

        Raises:
            ValueError: if manifest is invalid or missing required fields
            FileNotFoundError: if manifest file doesn't exist
        """
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

        data = toml.load(path)

        kata = data.get("kata")
        if not kata:
            raise ValueError("Missing [kata] section in manifest")

        # validate required fields
        required = ["name", "category", "base_difficulty", "description"]
        for field in required:
            if field not in kata:
                raise ValueError(f"Missing required field: {field}")

        # validate base_difficulty is an integer
        if not isinstance(kata["base_difficulty"], int):
            raise ValueError("base_difficulty must be an integer")

        # parse variations
        variations = []
        for var_data in data.get("variations", []):
            if "name" not in var_data:
                raise ValueError("Variation missing required field: name")

            variations.append(
                KataVariation(
                    name=var_data["name"],
                    description=var_data.get("description", ""),
                    params=var_data.get("params", {}),
                )
            )

        return cls(
            name=kata["name"],
            category=kata["category"],
            base_difficulty=kata["base_difficulty"],
            description=kata["description"],
            dependencies=kata.get("dependencies", []),
            variations=variations,
        )

    def validate(self, kata_dir: Path) -> None:
        """Validate that required files exist in kata directory.

        Args:
            kata_dir: path to kata directory

        Raises:
            FileNotFoundError: if required files are missing
        """
        required_files = ["template.py", "test_kata.py"]
        for filename in required_files:
            if not (kata_dir / filename).exists():
                raise FileNotFoundError(f"Missing required file: {filename}")


def validate_kata_directory(kata_dir: Path) -> KataManifest:
    """Validate a kata directory structure.

    Args:
        kata_dir: path to kata directory

    Returns:
        validated KataManifest

    Raises:
        FileNotFoundError: if manifest or required files missing
        ValueError: if manifest is invalid
    """
    manifest_path = kata_dir / "manifest.toml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.toml found in {kata_dir}")

    manifest = KataManifest.from_file(manifest_path)
    manifest.validate(kata_dir)

    return manifest


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python manifest_parser.py <kata_dir>")
        sys.exit(1)

    kata_dir = Path(sys.argv[1])
    try:
        manifest = validate_kata_directory(kata_dir)
        print(f"Valid kata: {manifest.name}")
        print(f"  Category: {manifest.category}")
        print(f"  Difficulty: {manifest.base_difficulty}")
        print(f"  Dependencies: {manifest.dependencies}")
        if manifest.variations:
            print(f"  Variations: {len(manifest.variations)}")
    except Exception as e:
        print(f"Invalid kata: {e}")
        sys.exit(1)
