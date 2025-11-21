# Workbook Quick Start Guide

## Overview

Workbooks are curated learning paths that group related exercises. They're defined as TOML manifests in `workbooks/<id>/manifest.toml` and generate HTML pages for browsing.

## Creating a New Workbook: Step-by-Step

### Step 1: Create the Manifest File

Create `workbooks/<workbook-id>/manifest.toml`:

```toml
[workbook]
id = "your_workbook_id"
title = "Workbook Title: Memorable Subtitle"
summary = "One sentence that describes what learners will gain."
learning_goals = [
  "Specific, measurable learning outcome 1",
  "Specific, measurable learning outcome 2",
  "Specific, measurable learning outcome 3",
]
prerequisites = ["kata_name_1", "kata_name_2"]  # Optional: katas to complete first
resources = [
  { title = "Official Documentation", url = "https://..." },
  { title = "Tutorial", url = "https://..." },
]
kata_namespace = "namespace"  # Directory prefix for all katas (e.g., "matplotlib", "plotly")

[[exercises]]
slug = "exercise-1-slug"
title = "Exercise 1 Title"
kata = "kata_name_matching_directory"
objective = "What the user will implement in one sentence."
acceptance = [
  "Concrete acceptance criterion 1",
  "Concrete acceptance criterion 2",
]
hints = [
  "Helpful direction without spoiling solution",
]
dependencies = []  # Other slugs in this workbook this depends on
assets = []  # Optional: ["images/example.png"]

[[exercises]]
slug = "exercise-2-slug"
title = "Exercise 2 Title"
kata = "kata_name_for_exercise_2"
objective = "What builds on Exercise 1."
acceptance = [
  "Criterion 1",
  "Criterion 2",
]
hints = [
  "Hint 1",
]
dependencies = ["exercise-1-slug"]  # Depends on first exercise
assets = []
```

### Step 2: Verify All Katas Exist

Before running anything, verify all referenced katas exist:

```bash
# Check if a kata exists
ls -la katas/exercises/matplotlib_scatter/manifest.toml

# List all katas in a namespace
ls katas/exercises/ | grep "^matplotlib_"
```

If a kata is missing, create it or update the manifest to reference an existing one.

### Step 3: Test in TUI

Build and run the project:

```bash
cargo build
cargo run

# In TUI: Press 'w' to open Workbooks view
# You should see your new workbook listed
```

### Step 4: (Optional) Create HTML Page

1. Copy the template:
```bash
cp assets/workbooks/_template/workbook-template.html \
   assets/workbooks/your_workbook_id/index.html
```

2. Edit the HTML with:
   - Introduction explaining the domain
   - Code comparisons (before/after patterns)
   - Learning context
   - Visual assets if applicable

3. Reference code snippets from kata templates:
```bash
cat katas/exercises/matplotlib_scatter/template.py
```

### Step 5: Commit

```bash
git add workbooks/your_workbook_id/manifest.toml
git commit -m "Add <workbook-name> workbook"
```

---

## Common Patterns

### Pattern 1: Foundation Building

**Ideal for:** Learning a single library or technique step-by-step

Example: Matplotlib Basics
1. Basic plot type (scatter)
2. Add styling (colors, markers)
3. Add labels and legends
4. Combine into layouts (subplots)

```toml
[[exercises]]
slug = "scatter"
# ...
dependencies = []

[[exercises]]
slug = "colors-markers"
# ...
dependencies = ["scatter"]

[[exercises]]
slug = "labels-titles"
# ...
dependencies = ["scatter"]

[[exercises]]
slug = "subplots"
# ...
dependencies = ["scatter", "labels-titles"]
```

### Pattern 2: Progressive Complexity

**Ideal for:** Algorithm or technique mastery

Example: Two Pointers Pattern
1. Simple case (valid palindrome)
2. Add constraint (sorted array)
3. Optimize further (container with most water)
4. Complex variant (trapping rain water)

```toml
[[exercises]]
slug = "palindrome"
# Base case: check if string is palindrome
dependencies = []

[[exercises]]
slug = "sorted-two-sum"
# Find pair in SORTED array
dependencies = ["palindrome"]

[[exercises]]
slug = "container-water"
# Find two lines with max area
dependencies = ["sorted-two-sum"]

[[exercises]]
slug = "trapping-water"
# Calculate water trapped (combines previous concepts)
dependencies = ["container-water"]
```

### Pattern 3: Integration

**Ideal for:** Combining separate concepts

Example: Einops Arena
1. Operation A (rearrange)
2. Operation B (reduce)
3. Operation C (einsum)
4. A + B + C together (attention logits)

```toml
[[exercises]]
slug = "patches"
kata = "einops_patches"
dependencies = []

[[exercises]]
slug = "segment-mean"
kata = "einops_segment_mean"
dependencies = ["patches"]

[[exercises]]
slug = "split-heads"
kata = "einops_split_heads"
dependencies = ["segment-mean"]

[[exercises]]
slug = "attention-logits"
kata = "einops_attention_logits"
dependencies = ["split-heads"]
```

---

## Exercise Design Rules

### Good Exercise Design

Each exercise should:
- Teach exactly ONE concept
- Be completable in 10-20 minutes
- Have clear input/output
- Include 2-3 concrete acceptance criteria
- Include 1-2 guiding hints (not the solution)

```toml
[[exercises]]
slug = "basic-plot"
title = "Create a Scatter Plot"
kata = "matplotlib_scatter"
objective = "Plot x-y coordinates with markers"
acceptance = [
  "Displays all data points",
  "X and Y axes are labeled",
]
hints = [
  "Use plt.scatter(x, y)",
  "Use plt.xlabel() and plt.ylabel()",
]
```

### Antipatterns to Avoid

**Too Vague:**
```toml
objective = "Learn about plotting"  # Bad: not specific
objective = "Create a scatter plot from x-y data"  # Good: specific and actionable
```

**Too Many Concepts:**
```toml
objective = "Create scatter plots with custom colors, markers, sizes, and labels"  # Bad: too much
objective = "Create a scatter plot with custom colors and markers"  # Better: focused
```

**Unclear Acceptance:**
```toml
acceptance = ["Plot looks good"]  # Bad: subjective
acceptance = ["All points displayed", "Colors match input array"]  # Good: concrete
```

---

## Directory Structure

```
kata-sr/
├── workbooks/
│   ├── matplotlib_basics/
│   │   └── manifest.toml          # Manifest file
│   ├── einops/
│   │   └── manifest.toml
│   └── your_workbook/
│       └── manifest.toml          # Create this
│
└── assets/workbooks/
    ├── matplotlib_basics/
    │   ├── index.html             # Generated HTML
    │   └── images/                # Optional: reference plots
    ├── einops_arena/
    │   └── index.html
    └── your_workbook/             # Optional: create this for custom HTML
        ├── index.html             # Optional: custom HTML page
        └── images/                # Optional: visual assets
```

---

## Full Example: Python Generators Workbook

```toml
[workbook]
id = "python_generators"
title = "Python Generators: Memory-Efficient Iteration"
summary = "Learn to write efficient, lazy iterators using yield and generator expressions."
learning_goals = [
  "Understand yield and generator protocol",
  "Write generators that produce values on demand",
  "Compose generators into pipelines",
  "Apply generators to real-world data processing",
]
prerequisites = []
resources = [
  { title = "Python Generators Documentation", url = "https://docs.python.org/3/howto/functional.html#generators" },
  { title = "RealPython: Generators", url = "https://realpython.com/generators/" },
]
kata_namespace = "python"

[[exercises]]
slug = "batch-generator"
title = "Simple Batch Generator"
kata = "python_batch_generator"
objective = "Implement a generator that yields fixed-size batches from a sequence."
acceptance = [
  "Yields exactly batch_size items per iteration",
  "Handles remaining items at end of sequence",
  "Uses O(batch_size) memory, not O(n)",
]
hints = [
  "Use yield inside a loop",
  "Slice the sequence in each iteration",
]
dependencies = []

[[exercises]]
slug = "cleanup"
title = "Generator Cleanup with Finally"
kata = "python_generator_cleanup"
objective = "Add cleanup logic to generators using try/finally."
acceptance = [
  "Cleanup code runs when generator exits normally",
  "Cleanup code runs when generator is garbage collected",
]
hints = [
  "Use try/finally inside the generator function",
  "Generator objects have a close() method",
]
dependencies = ["batch-generator"]

[[exercises]]
slug = "pipeline"
title = "Compose Generators into Pipelines"
kata = "python_generator_pipeline"
objective = "Chain multiple generators to build data processing pipelines."
acceptance = [
  "Each stage is a separate generator",
  "Data flows through stages lazily",
  "Entire pipeline uses minimal memory",
]
hints = [
  "Pass one generator to another as its input",
  "Each stage reads from the previous generator",
]
dependencies = ["cleanup"]

[[exercises]]
slug = "send-and-yield"
title = "Two-Way Communication with send()"
kata = "python_generator_send"
objective = "Use send() to pass data into running generators."
acceptance = [
  "Generator receives values via send()",
  "Received values are accessible after yield",
  "Handles initial None sent before first yield",
]
hints = [
  "Start generator with next() or send(None)",
  "Call send(value) to both receive output AND inject input",
]
dependencies = ["pipeline"]

[[exercises]]
slug = "yield-from"
title = "Delegating to Sub-generators with yield from"
kata = "python_yield_from"
objective = "Use yield from to delegate to sub-generators cleanly."
acceptance = [
  "All values from sub-generator are yielded",
  "Return value of sub-generator is accessible",
  "send() calls propagate to sub-generator",
]
hints = [
  "yield from replaces manual for/yield loops",
  "yield from <generator> returns the generator's return value",
]
dependencies = ["send-and-yield"]
```

---

## Troubleshooting

### Workbook doesn't appear in TUI

1. Check manifest syntax (validate TOML):
```bash
cargo build  # Will fail with parsing errors
```

2. Verify file path: `workbooks/<id>/manifest.toml` (exact case-sensitive spelling)

3. Check workbook id is unique (no duplicate ids across all workbooks)

### Exercise references missing kata

1. Verify kata exists:
```bash
ls katas/exercises/kata_name/manifest.toml
```

2. Verify kata name in manifest matches directory name exactly

3. Check `kata_namespace` setting - it prefixes all kata names:
```toml
kata_namespace = "matplotlib"
kata = "scatter"  # Resolves to: katas/exercises/matplotlib_scatter/
```

### Dependencies not working

1. Check all dependency slugs exist in workbook (case-sensitive)
2. Ensure no circular dependencies (A depends on B, B depends on A)
3. Verify dependencies field uses array syntax: `dependencies = ["slug1", "slug2"]`

### HTML page looks broken

1. Check image paths are relative to workbook directory:
```html
<!-- Wrong: absolute path -->
<img src="/assets/workbooks/matplotlib_basics/images/scatter.png">

<!-- Right: relative to workbook HTML -->
<img src="images/scatter.png">
```

2. Make sure HTML file is at: `assets/workbooks/<id>/index.html`

---

## Validation Before Commit

```bash
# 1. Check manifest is valid TOML
cargo build

# 2. Verify in TUI
cargo run
# Press 'w' -> navigate to your workbook
# Press 'a' to add exercises
# Press 'p' to preview first exercise

# 3. Check tests pass for all exercises
cd katas
uv run pytest exercises/ -v

# 4. Commit
git add workbooks/your_id/manifest.toml
git commit -m "Add <workbook-name> workbook"
```

---

## Resources

- **Workbook System Code:** `src/core/workbook.rs`, `src/tui/workbooks.rs`
- **Manifest Examples:** `workbooks/matplotlib_basics/manifest.toml`, `workbooks/einops/manifest.toml`
- **HTML Template:** `assets/workbooks/_template/workbook-template.html`
- **Full Analysis:** `WORKBOOK_ANALYSIS.md`

