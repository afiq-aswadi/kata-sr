# Kata-SR Workbook System Analysis

## Executive Summary

The kata-sr codebase has an established **workbook system** for creating curated learning paths that group related exercises. Two workbooks already exist (Matplotlib Basics and Einops Arena), and the system supports:

- Manifest-based workbook definitions (TOML format)
- HTML generation for browsing workbooks
- TUI integration for adding exercises to practice queue
- Progressive exercise sequencing with inter-exercise dependencies
- Visual assets and external learning resources

**Total Exercises in Database:** 289
**Existing Workbooks:** 2
**Potential Workbook Candidates:** Multiple domains identified

---

## 1. Current Workbook Implementation

### 1.1 Core Files

**Rust Implementation:**
- `/home/user/kata-sr/src/core/workbook.rs` - Workbook loading and HTML generation
- `/home/user/kata-sr/src/tui/workbooks.rs` - TUI interface for browsing workbooks

**Manifest & Assets:**
- `workbooks/*/manifest.toml` - Workbook definitions
- `assets/workbooks/<id>/index.html` - Generated HTML pages
- `assets/workbooks/_template/workbook-template.html` - HTML template
- `assets/workbooks/_template/README.md` - Template documentation

### 1.2 Workbook Structure

**Manifest Format (TOML):**
```toml
[workbook]
id = "unique_id"
title = "Display Title"
summary = "One-sentence summary"
learning_goals = ["Goal 1", "Goal 2"]
prerequisites = ["kata_name"]
resources = [
  { title = "Resource", url = "https://..." }
]
kata_namespace = "namespace"  # All exercises drawn from this namespace

[[exercises]]
slug = "exercise-slug"
title = "Exercise Title"
kata = "kata_name"  # Must exist in katas/exercises/
objective = "What you'll implement"
acceptance = ["Criterion 1", "Criterion 2"]
hints = ["Helpful hint"]
dependencies = ["other_exercise_slug"]  # Within workbook
assets = ["images/example.png"]  # Optional visual references
```

### 1.3 HTML Generation

- Reads kata templates (`katas/exercises/<kata_name>/template.py`)
- Generates styled HTML with dark theme (cyan accents)
- Includes exercise cards with objectives, acceptance criteria, hints
- Shows visual assets and code snippets
- Links to external learning resources

### 1.4 TUI Integration

**Workbooks Screen Actions:**
- `j/k` or arrow keys - Navigate workbook list
- `o` or Enter - Open workbook as HTML
- `a` - Add all exercises from workbook to practice queue
- `p` - Preview first exercise
- `Esc` - Back to main menu

---

## 2. Existing Workbooks

### 2.1 Matplotlib Basics

**Location:** `/home/user/kata-sr/workbooks/matplotlib_basics/manifest.toml`

**Meta:**
- ID: `matplotlib_basics`
- Title: "Matplotlib Basics: Visualization Fundamentals"
- Summary: "Build fluency with core matplotlib plotting patterns"
- 12 exercises (all currently available)
- Includes visual examples in `assets/workbooks/matplotlib_basics/images/`

**Exercise Progression:**
1. scatter - Basic scatter plot
2. line - Line plot with multiple series (depends: scatter)
3. bar - Bar chart (depends: scatter)
4. histogram - Histogram with bins (depends: bar)
5. labels-titles - Labels, titles, legends (depends: line)
6. colors-markers - Custom colors/markers (depends: scatter, labels-titles)
7. error-bars - Error bars (depends: line, labels-titles)
8. box-plot - Box plots (depends: bar, labels-titles)
9. heatmap - Heatmap with colorbar (depends: labels-titles)
10. contour - Contour plots (depends: heatmap)
11. subplots - Multiple subplots (depends: scatter, line, bar, histogram)
12. save-figure - Save figure to file (depends: labels-titles)

**Resources:**
- Matplotlib Documentation
- Pyplot Tutorial
- Gallery Examples

### 2.2 Einops Arena

**Location:** `/home/user/kata-sr/workbooks/einops/manifest.toml`

**Meta:**
- ID: `einops_arena`
- Title: "Einops Arena: Shape Fluency"
- Summary: "Build muscle memory for rearrange, reduce, and einsum"
- 4 exercises
- Prerequisites: `tensor_slicing`, `softmax`

**Exercise Progression:**
1. patches - Patchify images with rearrange
2. segment-mean - Sequence pooling with reduce (depends: patches)
3. split-heads - Unmerge attention heads (depends: segment-mean)
4. attention-logits - Scaled attention logits with einsum (depends: split-heads)

**Resources:**
- einops overview
- einops cheatsheet

---

## 3. Exercise Categories & Groupings

### 3.1 Category Distribution

| Category | Count | Type |
|----------|-------|------|
| exercises (NeetCode) | 151 | Algorithms & Data Structures |
| python | 25 | Python language features |
| interpretability | 20 | Transformer analysis & circuits |
| plotly | 13 | Interactive visualization |
| visualization | 12 | (generic visualization) |
| matplotlib | 10 | Matplotlib plotting |
| pytorch | 5 | PyTorch fundamentals |
| arrays | 6 | Array operations |
| strings | 5 | String algorithms |
| einops | 4 | Tensor reshaping |
| algorithms | 3 | General algorithms |
| Others | 9 | graphs, fundamentals, debug, etc. |
| **Total** | **289** | |

### 3.2 Detailed Category Breakdown

#### NeetCode / Algorithm Exercises (151 total)

These are LeetCode-style problems. No current natural grouping by topic in manifests.
Current structure: All tagged as category = "exercises"

**Sample problems:**
- neetcode_two_sum
- neetcode_3sum
- neetcode_valid_palindrome
- neetcode_binary_search
- neetcode_reverse_linked_list
- neetcode_valid_parentheses
- neetcode_trapping_rain_water
- ...and 144 more

#### Matplotlib (10 exercises)

All currently covered in `matplotlib_basics` workbook:
- matplotlib_scatter
- matplotlib_line
- matplotlib_bar
- matplotlib_histogram
- matplotlib_labels_titles
- matplotlib_colors_markers
- matplotlib_error_bars
- matplotlib_box_plot
- matplotlib_heatmap
- matplotlib_contour
- matplotlib_subplots
- matplotlib_save_figure
- matplotlib_animation_init
- matplotlib_arrow_annotation
- matplotlib_blitting
- matplotlib_custom_artist
- matplotlib_dual_axes
- matplotlib_figure_axes
- matplotlib_shape_highlight
- matplotlib_simple_animation
- matplotlib_span_highlight
- matplotlib_text_annotation

**Additional matplotlib exercises (not in workbook):**
- Animation & blitting
- Custom artists
- Advanced annotations
- Shape/span highlighting

#### Plotly (13 exercises)

All tagged as category = "plotly"

- plotly_scatter
- plotly_line
- plotly_bar
- plotly_layout
- plotly_hover_template
- plotly_hover_formatting
- plotly_multi_trace
- plotly_shared_axes
- plotly_grouped_bars
- plotly_dual_yaxis
- plotly_basic_grid
- plotly_spanning_subplot
- plotly_styled_markers
- plotly_update_subplot_axes
- plotly_subplot_annotation

#### Einops (4 exercises)

All currently covered in `einops_arena` workbook:
- einops_patches
- einops_segment_mean
- einops_split_heads
- einops_attention_logits

#### Python Language Features (25 exercises)

Grouped by topic within "python" category:

**Generators & Iteration:**
- python_batch_generator
- python_generator_cleanup
- python_generator_pipeline
- python_generator_send
- python_yield_from
- iterator_protocol

**Decorators & Functional:**
- memoize_decorator
- retry_decorator
- timing_decorator

**OOP & Classes:**
- create_simple_class
- create_class_with_init
- create_class_with_methods
- create_class_with_inheritance

**Metaclasses & Advanced:**
- auto_property_metaclass
- configurable_init_subclass
- context_manager
- descriptor_protocol
- lazy_property
- operator_overloading
- plugin_registration
- registry_metaclass
- singleton_metaclass
- typed_meta
- validated_init_subclass
- validated_meta

#### Interpretability (20 exercises)

Transformer mechanistic interpretability:
- activation_cache_pair
- ablate_attention_head
- activation_patching
- attention_head_patch
- basic_activation_patch
- circuit_analysis
- compare_attention_patterns
- compute_attention_entropy
- extract_attention_patterns
- find_previous_token_heads
- get_max_attention_positions
- logit_difference_metric
- patching_effect_metric
- residual_stream_patch
- systematic_head_scan
- transformerlens_extract_attention
- transformerlens_extract_residual
- transformerlens_position_indexing
- transformerlens_run_with_cache
- transformerlens_selective_cache

#### PyTorch Fundamentals (5 exercises)

- adam_optimizer
- batch_norm
- conv2d
- cross_entropy
- tensor_slicing

#### Arrays (6 exercises)

- find_missing_number
- rotate_array
- remove_duplicates
- tensor_slicing
- (others TBD)

#### Strings (5 exercises)

- string_is_palindrome
- string_reverse
- string_to_uppercase
- string_count_words
- string_join_with_separator

---

## 4. Database Schema for Workbooks

**Note:** Workbooks are NOT stored in the database. They are:
- Defined as manifest files in `workbooks/*/manifest.toml`
- Loaded on startup via `load_workbooks()` in `src/core/workbook.rs`
- Rendered to HTML
- Referenced in TUI

**Database Tables (from schema.rs):**
- `katas` - Exercise metadata (not workbook-specific)
- `kata_dependencies` - Prerequisite relationships
- `sessions` - Practice history
- `daily_stats` - Aggregated stats
- `kata_tags` - Tag system (can be used for workbook tagging)

---

## 5. Workbook Design Patterns

### Pattern 1: Foundation → Application
**Einops Arena example:**
1. Basic operation (patches)
2. Variation (segment-mean)
3. Combine skills (split-heads)
4. Real-world application (attention-logits)

### Pattern 2: Component Progression
**Matplotlib Basics example:**
1. Create single plot type
2. Style & customize
3. Add labels/legends
4. Combine into complex layouts

### Pattern 3: Atomic Focus
Each exercise teaches ONE concept:
- Don't mix attention computation with visualization
- Don't combine reshape + reduction in one exercise
- Use dependencies to chain related operations

---

## 6. Suggested Workbook Candidates

### 6.1 High Priority (ready to create)

#### Plotly Arena
- **Count:** 13 exercises
- **Structure:** Similar to Matplotlib but interactive
- **Progression:** Basic → Multi-trace → Advanced layouts → Subplots
- **Opportunity:** Currently no workbook for Plotly

#### Python Advanced Language Features
- **Count:** 25 exercises
- **Groupings:**
  - Generators & Iteration (5)
  - Decorators (3)
  - OOP & Classes (4)
  - Metaclasses & Advanced (13)
- **Opportunity:** Split into themed sub-workbooks

#### Interpretability & Circuits
- **Count:** 20 exercises
- **Groupings:**
  - Basic patching (activation_patching, basic_activation_patch, etc.)
  - TransformerLens integration (5 exercises)
  - Advanced analysis (entropy, metrics, circuits)
- **Opportunity:** Foundation for mechanistic interpretability

#### PyTorch Foundations
- **Count:** 5 exercises
- **Structure:** tensor_slicing → batch_norm → conv2d → adam → cross_entropy
- **Opportunity:** Entry point for ML practitioners

### 6.2 Medium Priority (need grouping)

#### NeetCode Algorithm Collections
- **Challenge:** 151 exercises, needs topical grouping
- **Possible Approaches:**
  1. **By Data Structure:** Arrays, Linked Lists, Trees, Graphs, Strings, Hash Maps, etc.
  2. **By Algorithm:** Sorting, Searching, DFS/BFS, Dynamic Programming, etc.
  3. **By Difficulty:** Easy, Medium, Hard
  4. **By Problem Pattern:** Two Pointers, Sliding Window, etc.
- **Opportunity:** Create 10-15 small focused workbooks instead of one monolithic one
- **Example:** "Two Pointers Pattern" (3-5 problems), "Binary Search Mastery" (4-6 problems)

### 6.3 Low Priority (wait for more exercises)

#### Advanced Visualization
- **Current:** Animation, Custom Artists, Advanced Annotations (5 matplotlib + plotly)
- **Status:** Too few for a standalone workbook yet
- **Opportunity:** Merge with Matplotlib Basics as "Advanced" section later

#### MLOps & Tools
- **Current:** wandb_basics, timing_decorator, activation_cache_pair (scattered)
- **Status:** Needs more exercises in this domain
- **Opportunity:** Create once tooling coverage improves

---

## 7. Implementation Checklist for New Workbooks

To create a new workbook:

1. **Define manifest** (`workbooks/<id>/manifest.toml`)
   - [ ] Unique workbook id
   - [ ] Clear title and summary
   - [ ] 3-5 learning goals
   - [ ] List prerequisites if any
   - [ ] Identify 4-12 related exercises
   - [ ] Define exercise progression with dependencies
   - [ ] Link to external resources

2. **Verify exercises exist**
   - [ ] All referenced katas exist in `katas/exercises/`
   - [ ] Each kata has template.py with sufficient content for reference

3. **Create HTML page** (optional but recommended)
   - [ ] Copy template: `assets/workbooks/_template/workbook-template.html`
   - [ ] Add introduction explaining the domain
   - [ ] Include code comparisons (before/after) if applicable
   - [ ] Write exercise summaries
   - [ ] Gather reference images if visual

4. **Add visual assets** (if applicable)
   - [ ] Create `assets/workbooks/<id>/images/` directory
   - [ ] Add reference plots/diagrams
   - [ ] Reference in manifest `assets` field
   - [ ] Embed in HTML with descriptive captions

5. **Test in TUI**
   - [ ] Run `cargo run` and navigate to Workbooks (press `w`)
   - [ ] Verify workbook appears
   - [ ] Test `a` to add exercises
   - [ ] Test `p` to preview first exercise
   - [ ] Test `o` to open HTML

6. **Documentation**
   - [ ] Add git commit with workbook manifest
   - [ ] If HTML is complex, add comments
   - [ ] Link to relevant issue/PR

---

## 8. Key Design Decisions

### Exercise Atomicity
Each workbook exercise should:
- Teach exactly ONE concept
- Have clear input/output
- Be completable in 10-20 minutes
- Not require knowledge of unreleased dependencies

### Dependency Management
- Use workbook `dependencies` field for exercise ordering
- Cross-workbook dependencies go in kata manifest `dependencies`
- Prerequisites listed in workbook `prerequisites` should be completed before starting

### Progressive Complexity
Good progression pattern:
```
Exercise 1: Concept A (standalone)
Exercise 2: Concept B (standalone)
Exercise 3: A + B together
Exercise 4: Add Concept C
Exercise 5: A + B + C + new technique
```

Bad progression:
```
Exercise 1: Concept A
Exercise 2: A + B + C + D (too much at once)
Exercise 3: Simple variant of A (should be earlier)
```

---

## 9. File Locations Summary

```
kata-sr/
├── src/
│   ├── core/workbook.rs          # Workbook loading & HTML generation
│   ├── tui/workbooks.rs          # TUI interface
│
├── workbooks/
│   ├── matplotlib_basics/
│   │   └── manifest.toml
│   ├── einops/
│   │   └── manifest.toml
│   └── [NEW WORKBOOKS HERE]
│
├── assets/workbooks/
│   ├── _template/
│   │   ├── workbook-template.html
│   │   └── README.md
│   ├── matplotlib_basics/
│   │   ├── index.html
│   │   └── images/
│   ├── einops_arena/
│   │   └── index.html
│   └── [NEW WORKBOOK HTML PAGES HERE]
│
├── katas/exercises/
│   ├── matplotlib_scatter/
│   ├── plotly_scatter/
│   ├── python_batch_generator/
│   ├── neetcode_two_sum/
│   └── [289 total exercise directories]
│
└── CLAUDE.md                      # Project overview
```

---

## 10. Summary Statistics

| Metric | Value |
|--------|-------|
| Total Exercises | 289 |
| Existing Workbooks | 2 |
| Existing Exercises in Workbooks | 16 (4 einops + 12 matplotlib) |
| Uncurated Exercises | 273 |
| Primary Opportunity: NeetCode | 151 exercises (52% of total) |
| Secondary Opportunity: Python | 25 exercises |
| Tertiary Opportunity: Visualizations | 23 exercises (10 matplotlib + 13 plotly) |
| Advanced ML Domain | 20 interpretability exercises |

---

## 11. Recommendations

### Immediate Actions (1-2 weeks)
1. **Create Plotly Arena workbook** (13 exercises, straightforward mapping to Matplotlib)
2. **Create Python Generators workbook** (5 focused exercises on yield/iteration)
3. **Test workbook discovery & HTML rendering** in TUI

### Short Term (2-4 weeks)
1. **Create NeetCode categories** - Pick 2-3 algorithm patterns, create focused workbooks
   - Example: "Two Pointers Pattern" (4-6 problems)
   - Example: "Binary Search Mastery" (4-5 problems)
   - Example: "Dynamic Programming Fundamentals" (5-6 problems)

2. **Create Python Advanced workbook** (Decorators, Metaclasses, OOP)

3. **Create PyTorch Fundamentals workbook** (5 exercises in order)

### Medium Term (1-3 months)
1. **Create Interpretability Foundation workbook** (20 exercises grouped by technique)
2. **Expand NeetCode coverage** - Build 10-15 focused algorithm workbooks
3. **Create advanced visualization workbook** (animations, custom artists)

### Validation Checklist
Before releasing any new workbook:
- [ ] All exercises pass their tests
- [ ] Dependencies are acyclic
- [ ] Learning progression is smooth
- [ ] HTML page renders correctly
- [ ] External resources are active
- [ ] Tested in TUI (add, preview, open)

---

## 12. Example: Creating "Two Pointers Pattern" Workbook

```toml
[workbook]
id = "two_pointers"
title = "Two Pointers Pattern: Efficient Array Solving"
summary = "Master the two-pointer technique for solving array and string problems efficiently."
learning_goals = [
  "Recognize when two-pointer approach applies",
  "Implement forward and backward pointer movement",
  "Solve problems with container, palindrome, and merge scenarios",
]
prerequisites = ["neetcode_two_sum"]
resources = [
  { title = "Two Pointers Explanation", url = "https://..." }
]
kata_namespace = "exercises"

[[exercises]]
slug = "valid-palindrome"
title = "Validate Palindrome with Two Pointers"
kata = "neetcode_valid_palindrome"
objective = "Check if a string is palindrome using two pointers from ends"
acceptance = [
  "Handles alphanumeric filtering correctly",
  "Case-insensitive comparison",
]
hints = ["Skip non-alphanumeric characters", "Compare from both ends"]
dependencies = []

[[exercises]]
slug = "two-sum-sorted"
title = "Two Sum in Sorted Array"
kata = "neetcode_two_sum_ii"
objective = "Find two indices in sorted array that sum to target"
acceptance = [
  "Returns 1-indexed array [index1, index2]",
  "Uses O(n) time and O(1) space",
]
hints = ["Move pointers inward based on sum comparison"]
dependencies = ["valid-palindrome"]

[[exercises]]
slug = "container-water"
title = "Container With Most Water"
kata = "neetcode_container_water"  # (hypothetical)
objective = "Find two lines that form container with max area"
acceptance = [
  "Returns maximum area possible",
  "Uses O(n) time complexity",
]
hints = ["Start at widest container", "Move pointer pointing to shorter line"]
dependencies = ["two-sum-sorted"]

[[exercises]]
slug = "trapping-rain"
title = "Trapping Rain Water"
kata = "neetcode_trapping_rain_water"
objective = "Calculate water trapped after rainfall"
acceptance = [
  "Handles edge cases with flat terrain",
  "Correct volume calculation",
]
hints = ["Track maximum heights on both sides"]
dependencies = ["container-water"]
```

---

## Conclusion

The workbook system is **well-designed and actively used**. With 289 exercises available and only 16 currently in workbooks, there's significant opportunity to:

1. Create focused learning paths for major domains (Plotly, Python, PyTorch)
2. Break down the 151 NeetCode exercises into 10-15 themed problem-solving workbooks
3. Create advanced paths for interpretability and ML concepts

The next step would be to **prioritize 3-5 workbooks** for immediate creation, then measure engagement to validate the progression and difficulty settings.

