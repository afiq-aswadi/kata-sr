# Workbook System File Reference

This document lists all absolute file paths related to the workbook system for easy navigation.

## Analysis Documents (Created by analysis)

These files provide comprehensive information about the workbook system:

- `/home/user/kata-sr/WORKBOOK_ANALYSIS.md` (657 lines, 20KB)
  Complete analysis of workbook system, exercise categorization, design patterns

- `/home/user/kata-sr/WORKBOOK_QUICK_START.md` (469 lines, 12KB)
  Step-by-step guide to creating new workbooks with examples

- `/home/user/kata-sr/WORKBOOK_IMPLEMENTATION.md` (577 lines, 15KB)
  Technical architecture, code structure, and integration details

- `/home/user/kata-sr/WORKBOOK_FILE_REFERENCE.md` (This file)
  Map of all workbook-related file paths

---

## Core Rust Implementation

### Workbook System Core

- `/home/user/kata-sr/src/core/workbook.rs` (507 lines)
  - `load_workbooks()` - Load all workbooks from filesystem
  - `load_manifest()` - Parse workbook TOML manifest
  - `validate_workbook()` - Validate manifest structure
  - `generate_workbook_html()` - Render HTML from manifest
  - `render_html()` - Create styled HTML page
  - `render_exercise()` - Create exercise card HTML
  - `load_template_snippet()` - Extract code preview from kata template

### Workbook TUI Screen

- `/home/user/kata-sr/src/tui/workbooks.rs` (264 lines)
  - `WorkbookScreen` - TUI navigation interface
  - `WorkbookAction` - User actions (navigate, add, preview, open)
  - `render_list()` - Draw workbook list pane
  - `render_detail()` - Draw details pane
  - `render_footer()` - Draw keyboard help footer
  - `handle_input()` - Process keyboard input

### Related Systems

- `/home/user/kata-sr/src/core/kata_loader.rs`
  - `load_available_katas()` - Load all exercise metadata
  - Used by workbook system for validation and reference

- `/home/user/kata-sr/src/db/schema.rs`
  - Database tables: `katas`, `kata_dependencies`, `sessions`, `daily_stats`
  - Note: Workbooks are NOT stored in database

- `/home/user/kata-sr/src/main.rs`
  - Application entry point
  - Calls workbook loading when needed

---

## Workbook Manifests

### Existing Workbooks

- `/home/user/kata-sr/workbooks/matplotlib_basics/manifest.toml`
  - 12 exercises: scatter, line, bar, histogram, labels-titles, etc.
  - Complete example of well-structured workbook

- `/home/user/kata-sr/workbooks/einops/manifest.toml`
  - 4 exercises: patches, segment-mean, split-heads, attention-logits
  - Example of integration/arena pattern

### Template for New Workbooks

Create new workbooks at:
```
/home/user/kata-sr/workbooks/<workbook_id>/manifest.toml
```

Example path:
```
/home/user/kata-sr/workbooks/plotly_arena/manifest.toml
/home/user/kata-sr/workbooks/python_generators/manifest.toml
/home/user/kata-sr/workbooks/two_pointers/manifest.toml
```

---

## HTML Output

### Generated Workbook Pages

- `/home/user/kata-sr/assets/workbooks/matplotlib_basics/index.html`
  - Generated from matplotlib_basics manifest
  - Includes 12 exercise cards with code snippets

- `/home/user/kata-sr/assets/workbooks/einops_arena/index.html`
  - Generated from einops manifest
  - 4 exercise cards

### Visual Assets

- `/home/user/kata-sr/assets/workbooks/matplotlib_basics/images/`
  - scatter_example.png
  - line_example.png
  - bar_example.png
  - histogram_example.png
  - ... and more visual references

### HTML Template

- `/home/user/kata-sr/assets/workbooks/_template/workbook-template.html`
  - Base HTML template for workbooks
  - Styled with dark theme, cyan accents
  - Responsive design

- `/home/user/kata-sr/assets/workbooks/_template/README.md`
  - Documentation for HTML template
  - Placeholder tokens for customization

---

## Exercise Directories

All exercises are located in:

```
/home/user/kata-sr/katas/exercises/
```

### By Category

**Matplotlib (10 exercises):**
```
/home/user/kata-sr/katas/exercises/matplotlib_scatter/
/home/user/kata-sr/katas/exercises/matplotlib_line/
/home/user/kata-sr/katas/exercises/matplotlib_bar/
/home/user/kata-sr/katas/exercises/matplotlib_histogram/
/home/user/kata-sr/katas/exercises/matplotlib_labels_titles/
/home/user/kata-sr/katas/exercises/matplotlib_colors_markers/
/home/user/kata-sr/katas/exercises/matplotlib_error_bars/
/home/user/kata-sr/katas/exercises/matplotlib_box_plot/
/home/user/kata-sr/katas/exercises/matplotlib_heatmap/
/home/user/kata-sr/katas/exercises/matplotlib_contour/
/home/user/kata-sr/katas/exercises/matplotlib_subplots/
/home/user/kata-sr/katas/exercises/matplotlib_save_figure/
... (22 total matplotlib exercises)
```

**Plotly (13 exercises):**
```
/home/user/kata-sr/katas/exercises/plotly_scatter/
/home/user/kata-sr/katas/exercises/plotly_line/
/home/user/kata-sr/katas/exercises/plotly_bar/
... (13 total plotly exercises)
```

**Einops (4 exercises):**
```
/home/user/kata-sr/katas/exercises/einops_patches/
/home/user/kata-sr/katas/exercises/einops_segment_mean/
/home/user/kata-sr/katas/exercises/einops_split_heads/
/home/user/kata-sr/katas/exercises/einops_attention_logits/
```

**Python (25 exercises):**
```
/home/user/kata-sr/katas/exercises/python_batch_generator/
/home/user/kata-sr/katas/exercises/python_generator_cleanup/
/home/user/kata-sr/katas/exercises/python_generator_pipeline/
/home/user/kata-sr/katas/exercises/python_generator_send/
/home/user/kata-sr/katas/exercises/python_yield_from/
/home/user/kata-sr/katas/exercises/create_simple_class/
/home/user/kata-sr/katas/exercises/create_class_with_init/
... (25 total python exercises)
```

**NeetCode / Algorithms (151 exercises):**
```
/home/user/kata-sr/katas/exercises/neetcode_two_sum/
/home/user/kata-sr/katas/exercises/neetcode_3sum/
/home/user/kata-sr/katas/exercises/neetcode_valid_palindrome/
/home/user/kata-sr/katas/exercises/neetcode_binary_search/
... (151 total algorithm exercises)
```

**Interpretability (20 exercises):**
```
/home/user/kata-sr/katas/exercises/activation_cache_pair/
/home/user/kata-sr/katas/exercises/ablate_attention_head/
/home/user/kata-sr/katas/exercises/activation_patching/
/home/user/kata-sr/katas/exercises/transformerlens_run_with_cache/
... (20 total interpretability exercises)
```

**PyTorch (5 exercises):**
```
/home/user/kata-sr/katas/exercises/tensor_slicing/
/home/user/kata-sr/katas/exercises/batch_norm/
/home/user/kata-sr/katas/exercises/conv2d/
/home/user/kata-sr/katas/exercises/adam_optimizer/
/home/user/kata-sr/katas/exercises/cross_entropy/
```

**Other:**
```
/home/user/kata-sr/katas/exercises/arrays/
/home/user/kata-sr/katas/exercises/strings/
/home/user/kata-sr/katas/exercises/graphs/
... (various others)
```

### Exercise Structure (Each has):

```
/home/user/kata-sr/katas/exercises/<exercise_name>/
├── manifest.toml        # Metadata and dependencies
├── template.py          # Starter code with TODOs
├── reference.py         # Solution (for reference)
└── test_kata.py         # Pytest tests
```

---

## Database Schema

Location: `/home/user/kata-sr/src/db/schema.rs`

**Important Note:** Workbooks are NOT stored in the database.

Tables relevant to workbooks:
- `katas` - Individual exercise metadata (name, category, description)
- `kata_dependencies` - Prerequisites between exercises
- `kata_tags` - Flexible tagging system
- `sessions` - Practice history (per exercise, not per workbook)
- `daily_stats` - Aggregate statistics

---

## Configuration Files

Project configuration:
- `/home/user/kata-sr/Cargo.toml` - Rust project manifest
- `/home/user/kata-sr/Cargo.lock` - Dependency lock file

Python environment:
- `/home/user/kata-sr/katas/pyproject.toml` - Python dependencies (uv)
- `/home/user/kata-sr/katas/.venv/` - Virtual environment (created at runtime)

---

## Documentation Files

Project documentation:
- `/home/user/kata-sr/CLAUDE.md` - Project overview and vision
- `/home/user/kata-sr/README.md` - Installation and usage instructions
- `/home/user/kata-sr/AGENT_1.md` - Rust core & database guide
- `/home/user/kata-sr/AGENT_2.md` - Python framework guide
- `/home/user/kata-sr/AGENT_3.md` - TUI application guide
- `/home/user/kata-sr/AGENT_4.md` - Example katas guide
- `/home/user/kata-sr/AGENT_5.md` - Analytics & integration guide

Workbook documentation (created by analysis):
- `/home/user/kata-sr/WORKBOOK_ANALYSIS.md` - System analysis
- `/home/user/kata-sr/WORKBOOK_QUICK_START.md` - Creation guide
- `/home/user/kata-sr/WORKBOOK_IMPLEMENTATION.md` - Technical reference
- `/home/user/kata-sr/WORKBOOK_FILE_REFERENCE.md` - This file

---

## Key Paths Summary

| Purpose | Path |
|---------|------|
| **Workbook Manifests** | `workbooks/<id>/manifest.toml` |
| **Generated HTML** | `assets/workbooks/<id>/index.html` |
| **Exercise Directories** | `katas/exercises/<exercise_name>/` |
| **Template Snippets** | `katas/exercises/<exercise_name>/template.py` |
| **Rust Code** | `src/core/workbook.rs`, `src/tui/workbooks.rs` |
| **Database** | `~/.local/share/kata-sr/kata.db` (user data) |
| **Analysis Docs** | `WORKBOOK_*.md` (in project root) |

---

## Navigation Quick Links

When creating a new workbook, you'll typically need to:

1. Check existing workbooks for patterns:
   - `/home/user/kata-sr/workbooks/matplotlib_basics/manifest.toml`
   - `/home/user/kata-sr/workbooks/einops/manifest.toml`

2. Verify exercises exist:
   - `/home/user/kata-sr/katas/exercises/your_exercise/manifest.toml`

3. Create your workbook manifest:
   - `/home/user/kata-sr/workbooks/your_id/manifest.toml`

4. (Optional) Create custom HTML:
   - `/home/user/kata-sr/assets/workbooks/your_id/index.html`

5. Test in TUI:
   - Run `cargo run` from project root
   - Press 'w' to view workbooks

6. Review code architecture:
   - `/home/user/kata-sr/src/core/workbook.rs` - Main logic
   - `/home/user/kata-sr/src/tui/workbooks.rs` - UI implementation

---

## Database Location

User's local database:
```
~/.local/share/kata-sr/kata.db
```

The workbook system reads manifests from disk (not database) and stores practice history in this database.

---

## Total Exercise Count by Directory

```bash
# Count all exercises
ls -d /home/user/kata-sr/katas/exercises/*/ | wc -l
# Result: 289

# Count by category (see WORKBOOK_ANALYSIS.md for details)
find /home/user/kata-sr/katas/exercises -name "manifest.toml" \
  -exec grep "^category = " {} \; | wc -l
# Result: 289

# Exercises in workbooks
# matplotlib_basics: 12
# einops_arena: 4
# Total in workbooks: 16
# Uncurated: 273
```

---

For more details, see:
- `WORKBOOK_ANALYSIS.md` - Complete system analysis
- `WORKBOOK_QUICK_START.md` - How to create workbooks
- `WORKBOOK_IMPLEMENTATION.md` - Technical deep dive

