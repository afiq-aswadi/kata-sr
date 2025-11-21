# Workbook System Implementation Details

This document provides technical details about how the workbook system works internally.

## Code Architecture

### 1. Core Workbook Module

**File:** `/home/user/kata-sr/src/core/workbook.rs` (507 lines)

#### Key Data Structures

```rust
// Top-level workbook metadata from manifest
pub struct WorkbookMeta {
    pub id: String,
    pub title: String,
    pub summary: String,
    pub learning_goals: Vec<String>,
    pub prerequisites: Vec<String>,
    pub resources: Vec<WorkbookResource>,
    pub kata_namespace: Option<String>,
}

// Exercise within a workbook
pub struct WorkbookExercise {
    pub slug: String,
    pub title: String,
    pub kata: String,                // Reference to actual kata in katas/exercises/
    pub objective: String,
    pub acceptance: Vec<String>,
    pub hints: Vec<String>,
    pub assets: Vec<String>,         // Optional: image paths
    pub dependencies: Vec<String>,   // Depends on other exercise slugs
}

// In-memory workbook representation
pub struct Workbook {
    pub meta: WorkbookMeta,
    pub exercises: Vec<WorkbookExercise>,
    pub manifest_path: PathBuf,      // workbooks/<id>/manifest.toml
    pub html_path: PathBuf,          // assets/workbooks/<id>/index.html
}
```

#### Key Functions

```rust
// Load all workbooks from workbooks/ directory
pub fn load_workbooks() -> Result<Vec<Workbook>>

// Validate workbook structure
fn validate_workbook(
    manifest: &WorkbookManifest,
    katas_by_name: &HashMap<String, kata_loader::AvailableKata>,
) -> Result<()>

// Generate HTML from workbook definition
pub fn generate_workbook_html(workbook: &Workbook) -> Result<()>

// Render workbook metadata to HTML
fn render_html(workbook: &Workbook, snippets: &[Option<String>]) -> String

// Render individual exercise card
fn render_exercise(idx: usize, ex: &WorkbookExercise, snippet: Option<String>) -> String

// Load template.py snippet for display in HTML
fn load_template_snippet(kata_name: &str) -> Result<String>
```

#### Validation Rules

The `validate_workbook()` function checks:

1. **Workbook Metadata:**
   - Non-empty id, title
   - At least one exercise

2. **Exercise Validity:**
   - Non-empty slug, title, objective
   - Referenced kata exists in katas_by_name
   - No duplicate exercise slugs

3. **Dependency Validity:**
   - Exercise dependencies reference existing slugs
   - No forward references or cycles (not checked, but should be acyclic by design)

#### HTML Generation

The workbook system reads:
- Manifest: `workbooks/<id>/manifest.toml` (TOML metadata)
- Template snippet: `katas/exercises/<kata_name>/template.py` (code preview, truncated to 80 lines)

And generates:
- HTML: `assets/workbooks/<id>/index.html` (styled page with exercise cards)

The HTML includes:
- Workbook title and summary
- Learning goals (bulleted list)
- Prerequisites (pill-styled tags)
- External resources (links)
- Exercise cards with:
  - Exercise number, slug, kata name
  - Title and objective
  - Acceptance criteria (if any)
  - Hints (if any)
  - Asset images (if any)
  - Template code snippet (if available)
  - Exercise dependencies (pill tags)

---

### 2. TUI Workbooks Screen

**File:** `/home/user/kata-sr/src/tui/workbooks.rs` (264 lines)

#### Data Structure

```rust
pub struct WorkbookScreen {
    workbooks: Vec<Workbook>,                           // Loaded workbooks
    available_by_name: HashMap<String, AvailableKata>, // All exercises
    selected: usize,                                    // Currently selected workbook
    list_state: ListState,                             // Ratatui list state
}

pub enum WorkbookAction {
    None,                                   // No action
    Back,                                   // Return to main screen
    OpenHtml(PathBuf),                      // Open workbook HTML
    AddExercises {
        kata_names: Vec<String>,            // Exercises to add
        workbook_title: String,
    },
    PreviewFirst(AvailableKata),           // Preview first exercise
}
```

#### Keyboard Bindings

| Key | Action |
|-----|--------|
| `j` / Down | Select next workbook |
| `k` / Up | Select previous workbook |
| `o` / Enter | Open workbook HTML (opens in default browser) |
| `a` | Add all exercises from workbook to practice queue |
| `p` | Preview first exercise (jump to it in library view) |
| `Esc` | Return to main menu |

#### Rendering

The screen is divided into two panes:

1. **Left (40%):** List of workbooks with exercise count
   - Cyan highlight for selected
   - Shows: "Title · N exercises"

2. **Right (60%):** Details of selected workbook
   - Title
   - Summary
   - Learning goals (bulleted)
   - Exercise list (with objective preview)
   - Resources (if any)

#### Integration Points

- `WorkbookScreen::load()` calls `load_workbooks()` from `src/core/workbook.rs`
- Returns `WorkbookAction` to main app for navigation
- Adds exercises via app's library integration

---

## Workbook Manifest Format

**Location:** `workbooks/<id>/manifest.toml`

**TOML Structure:**

```toml
[workbook]
id = "unique_id"
title = "Display Title"
summary = "Short description"
learning_goals = ["goal 1", "goal 2"]
prerequisites = ["kata_id"]
resources = [
  { title = "Resource Name", url = "https://..." }
]
kata_namespace = "namespace"

[[exercises]]
slug = "exercise-slug"
title = "Exercise Title"
kata = "kata_name"
objective = "What you'll implement"
acceptance = ["criterion 1", "criterion 2"]
hints = ["hint 1"]
dependencies = ["previous-slug"]
assets = ["images/example.png"]
```

**Parsing:**

The manifest is parsed using `serde` with `toml::from_str()`.

```rust
#[derive(Debug, Deserialize)]
struct WorkbookManifest {
    workbook: WorkbookMeta,
    exercises: Vec<WorkbookExercise>,
}
```

All fields use `#[serde(default)]` for optional fields except where explicitly required.

---

## File Locations

### Reading Workbooks

```
workbooks/
├── matplotlib_basics/
│   └── manifest.toml            // Read by load_workbooks()
├── einops/
│   └── manifest.toml
└── [new workbook]/
    └── manifest.toml
```

### Writing HTML Output

```
assets/workbooks/
├── matplotlib_basics/
│   ├── index.html               // Generated by generate_workbook_html()
│   └── images/                  // Reference plots (optional)
│       ├── scatter_example.png
│       ├── line_example.png
│       └── ...
├── einops_arena/
│   └── index.html
└── [new workbook]/
    ├── index.html               // Generated or manually created
    └── images/                  // Optional visual assets
        └── *.png
```

### Exercise References

```
katas/exercises/
├── matplotlib_scatter/
│   ├── manifest.toml
│   ├── template.py              // Read by load_template_snippet()
│   ├── reference.py
│   └── test_kata.py
├── einops_patches/
│   ├── manifest.toml
│   ├── template.py
│   ├── reference.py
│   └── test_kata.py
└── [referenced katas]/
    └── template.py
```

---

## Workbook Loading Pipeline

### Startup

1. **Application Start** (`main.rs`)
   - Initializes database
   - Calls `load_workbooks()` when navigating to workbooks view

2. **Load Workbooks** (`src/core/workbook.rs::load_workbooks()`)
   - Scans `workbooks/` directory
   - For each subdirectory with `manifest.toml`:
     - Loads and parses manifest
     - Validates against available katas
     - Checks for duplicate IDs
     - Creates `Workbook` struct
   - Returns `Vec<Workbook>`

3. **TUI Screen** (`src/tui/workbooks.rs::WorkbookScreen::load()`)
   - Receives `Vec<Workbook>`
   - Loads all available katas for reference
   - Initializes UI state
   - Selects first workbook

4. **User Navigation**
   - Select workbook
   - Press `o` to open HTML or `a` to add exercises
   - TUI handles action (open browser / add to queue)

### HTML Generation

Called when:
- Workbook manifest is created/updated
- User requests workbook regeneration
- Build system runs workbook generation step

Process:
1. Read manifest from `workbooks/<id>/manifest.toml`
2. For each exercise in manifest:
   - Try to load `katas/exercises/<kata_name>/template.py`
   - Truncate to first 80 lines
   - Store as code snippet (graceful failure if not found)
3. Render HTML template with:
   - Metadata (title, goals, resources)
   - Exercise cards with snippets
   - Styling (dark theme, cyan accents)
4. Write to `assets/workbooks/<id>/index.html`

---

## Database Interaction

**Important:** Workbooks are NOT stored in the database.

However, when exercises are added to the practice queue:
- Workbook exercises are referenced by kata name
- Katas are loaded from `katas/` manifest and stored in `katas` table
- Sessions table tracks practice history
- FSRS state is updated per kata (not per workbook)

**Future Enhancement:** Could add workbook_id column to sessions table to track which workbook a practice session came from.

---

## Integration with Other Systems

### 1. Kata Loader

**File:** `src/core/kata_loader.rs`

```rust
pub fn load_available_katas() -> Result<Vec<AvailableKata>>
```

Used by workbook system to:
- Validate exercise references
- Build `available_by_name` map in TUI
- Create `AvailableKata` objects for preview

### 2. Library/Catalog

**File:** `src/tui/library.rs`

When user presses `a` to add all exercises from a workbook:
- `WorkbookAction::AddExercises` is returned
- App adds each exercise to library by name
- Library loads full kata from `katas/exercises/` directory

### 3. Practice Flow

When user selects an exercise:
1. **TUI selects exercise** (individual kata from library, may have come from workbook)
2. **Runner spawns test process** (JSON over stdio to Python)
3. **Results are shown** (pass/fail, test output)
4. **FSRS rating** is applied (individual kata scheduling, not workbook-level)

---

## Extensibility

### Adding Workbook Features

The current design supports:
- Static metadata (title, goals, summary, resources)
- Exercise progression with dependencies
- Optional visual assets
- External learning resources

Potential future additions:
- Completion tracking (all exercises in workbook completed?)
- Custom workbook CSS themes
- Video embeddings (via assets system)
- Interactive quizzes between exercises
- Progress percentage display in TUI
- Estimated time to complete per exercise
- Difficulty metadata (easy/medium/hard)
- Community user reviews/ratings

### Modularity

Each workbook is independent:
- Manifest is self-contained
- HTML can be customized per workbook
- No cross-workbook dependencies (by design)
- Easy to add/remove/update without affecting others

---

## Performance Considerations

### Workbook Loading

- **Startup cost:** O(n) where n = number of workbooks (typically <20)
- **Manifest parsing:** O(1) per workbook (small TOML files)
- **Validation:** O(m) where m = exercises in workbook (typically <20)
- **Kata lookup:** O(k) where k = total available katas (cached in HashMap)

### HTML Generation

- **Template reading:** O(e) where e = exercises (read template.py snippets)
- **Rendering:** O(e) string concatenation (not critical path)
- **File I/O:** O(1) write operation

### TUI Rendering

- **List drawing:** O(w) where w = visible workbooks (typically 5-10)
- **Detail pane:** O(e) for exercise list (typically <20)
- **No database queries** - all in-memory

---

## Code Quality

### Testing

**Note:** Workbook system doesn't have dedicated unit tests yet.

Recommended tests to add:
1. `test_load_valid_manifest()`
2. `test_detect_missing_kata()`
3. `test_detect_duplicate_ids()`
4. `test_detect_circular_dependencies()`
5. `test_html_generation()`
6. `test_tui_navigation()`

### Type Safety

- Full Rust type checking
- Serde validation during deserialization
- Path safety with `PathBuf`
- Result types for error handling

### Error Handling

Current errors:
- Missing manifest file → `anyhow!`
- Invalid TOML → parsed error message
- Missing kata reference → validation error
- Duplicate workbook id → validation error

All errors bubble up to caller with context.

---

## Related Configuration

### Environment Variables

`load_template_snippet()` respects:
```rust
KATA_SR_KATAS_DIR  // Override default "katas" directory
```

### Cargo Features

None currently - workbook system is always compiled.

---

## Common Implementation Patterns

### Pattern: Manifest with Sections

The workbook manifest uses TOML `[table]` and `[[array of tables]]`:

```toml
[workbook]          # Single section
id = "..."

[[exercises]]       # Array of exercise objects
slug = "..."
```

This is the idiomatic TOML pattern and matches similar systems.

### Pattern: Validation Before Use

All loaded data is validated before use:
```rust
let manifest = load_manifest(&manifest_path)?;
validate_workbook(&manifest, &katas_by_name)?;
```

This ensures invalid data never reaches the TUI.

### Pattern: Graceful Degradation

When reading template snippets:
```rust
.map(|ex| load_template_snippet(&ex.kata).ok())
```

Returns `None` if snippet not found, page still renders without it.

---

## Debugging

### Enable Workbook Logging

Add debug print in `load_workbooks()`:
```rust
eprintln!("Loaded {} workbooks", workbooks.len());
for wb in &workbooks {
    eprintln!("  - {} ({})", wb.meta.title, wb.meta.id);
    for ex in &wb.exercises {
        eprintln!("    - {} ({})", ex.title, ex.slug);
    }
}
```

### Validate Manifest Syntax

```bash
# Run cargo build - will show TOML parse errors
cargo build

# Or test directly:
cd workbooks/your_id
cat manifest.toml | python3 -m tomllib
```

### Check HTML Output

```bash
# After running app, check generated HTML:
cat assets/workbooks/your_id/index.html | head -50
```

### Test Exercise References

```bash
# Verify all referenced katas exist:
for kata in $(grep 'kata = ' workbooks/*/manifest.toml); do
    ls katas/exercises/$kata/manifest.toml 2>/dev/null || echo "Missing: $kata"
done
```

---

## Future Enhancement Ideas

1. **Workbook Completion Tracking**
   - Add `workbooks_progress` table to database
   - Track which workbooks user has started/completed

2. **Adaptive Difficulty**
   - Detect if user is struggling (success rate < 50%)
   - Suggest easier workbooks or prerequisites

3. **Multi-Language Support**
   - Translate learning goals and exercise descriptions
   - Bundle language in workbook manifest

4. **Collaborative Features**
   - Share custom workbooks via git
   - Community rating/review system
   - Discussion threads per exercise

5. **Rich Media**
   - Embedded videos in HTML pages
   - Interactive code sandboxes
   - Visualization of problem progression

6. **Analytics**
   - Time-to-complete per workbook
   - Success rate by exercise
   - Difficulty calibration based on user data

