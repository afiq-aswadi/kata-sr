"""Extract and validate TODO/BLANK markers in kata templates."""

import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BlankRegion:
    """Represents a BLANK region in a template."""

    start_line: int
    end_line: int
    content: str


def extract_blanks(template_path: Path) -> list[BlankRegion]:
    """Extract BLANK_START/BLANK_END regions from template.

    Args:
        template_path: path to template file

    Returns:
        list of blank regions with line numbers
    """
    lines = template_path.read_text().splitlines()
    blanks = []
    current_blank: dict[str, int | list[str]] | None = None

    for i, line in enumerate(lines, start=1):
        if "# BLANK_START" in line or "# TODO" in line:
            if current_blank is None:
                current_blank = {"start": i, "lines": []}
        elif "# BLANK_END" in line:
            if current_blank is not None:
                blanks.append(
                    BlankRegion(
                        start_line=current_blank["start"],  # type: ignore
                        end_line=i,
                        content="\n".join(current_blank["lines"]),  # type: ignore
                    )
                )
                current_blank = None
        elif current_blank is not None:
            current_blank["lines"].append(line)  # type: ignore

    return blanks


def count_todos(template_path: Path) -> int:
    """Count TODO markers in template.

    Args:
        template_path: path to template file

    Returns:
        number of TODO markers found
    """
    content = template_path.read_text()
    return len(re.findall(r"#\s*TODO", content))


def is_template_filled(template_path: Path) -> bool:
    """Check if template has been filled in by user.

    Heuristic: no remaining TODO markers and BLANK regions have content.

    Args:
        template_path: path to template file

    Returns:
        True if template appears to be filled, False otherwise
    """
    if count_todos(template_path) > 0:
        return False

    blanks = extract_blanks(template_path)
    for blank in blanks:
        # check if blank region is not just "pass" or empty
        content = blank.content.strip()
        if not content or content == "pass":
            return False

    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python template_validator.py <template.py>")
        sys.exit(1)

    template = Path(sys.argv[1])
    blanks = extract_blanks(template)
    todos = count_todos(template)

    print(f"Template: {template.name}")
    print(f"TODO markers: {todos}")
    print(f"BLANK regions: {len(blanks)}")
    for blank in blanks:
        print(f"  Lines {blank.start_line}-{blank.end_line}")

    if is_template_filled(template):
        print("Template appears to be filled")
    else:
        print("Template needs work")
