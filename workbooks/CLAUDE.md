# Workbooks Guide

## What are Workbooks?

Workbooks are curated learning paths that group related katas into a cohesive educational experience. Unlike individual katas which focus on atomic skills, workbooks provide:

- **Context and motivation**: Why these skills matter
- **Progressive structure**: Exercises build on each other
- **Educational content**: Explanations, code comparisons, and examples
- **Curated resources**: Links to relevant documentation and tutorials

Think of them as mini-courses or "arenas" where you develop fluency in a specific domain.

## File Structure

```
workbooks/
├── CLAUDE.md                           # This file
├── <workbook-namespace>/
│   └── manifest.toml                   # Workbook metadata
└── ../assets/workbooks/
    ├── _template/
    │   ├── workbook-template.html      # HTML template
    │   └── README.md                   # Template usage guide
    └── <workbook-namespace>/
        ├── index.html                  # The workbook page
        └── images/                     # (Optional) Reference plots/diagrams
            └── *.png
```

**Important**: The manifest lives in `workbooks/` but the HTML page lives in `assets/workbooks/`. This separation keeps kata metadata close to kata files while keeping web assets organized.

## Creating a New Workbook

### Step 1: Define the Manifest

Create `workbooks/<namespace>/manifest.toml`:

```toml
[workbook]
id = "unique_workbook_id"
title = "Display Title"
summary = "One-sentence description of what this workbook teaches."
learning_goals = [
  "Specific skill or understanding goal",
  "Another concrete learning outcome",
]
prerequisites = ["kata_id_1", "kata_id_2"]  # Katas that should be mastered first
resources = [
  { title = "Resource Name", url = "https://..." },
]
kata_namespace = "namespace"  # All exercises draw from katas/<namespace>/

[[exercises]]
slug = "url-friendly-name"
title = "Human-Readable Exercise Title"
kata = "kata_id"  # Must exist in katas/<namespace>/
objective = "What you'll implement in one sentence."
acceptance = [
  "Concrete criterion 1",
  "Concrete criterion 2",
]
hints = [
  "Helpful hint without giving away the solution",
]
dependencies = ["other_exercise_slug"]  # Within this workbook
assets = []  # Optional: ["images/example.png"] for reference plots/diagrams
```

### Step 2: Create the HTML Page

1. Copy the template:
   ```bash
   cp assets/workbooks/_template/workbook-template.html \
      assets/workbooks/<namespace>/index.html
   ```

2. Replace all `{{PLACEHOLDER}}` tokens with your content. See `assets/workbooks/_template/README.md` for the complete list.

3. Read the kata templates to populate code examples:
   ```bash
   cat katas/exercises/<kata_name>/template.py
   ```

### Step 3: Write Educational Content

The introduction should answer:
- **What**: What skill/library/pattern does this teach?
- **Why**: Why is this important? What problems does it solve?
- **How**: Brief overview of the approach or mental model

Include:
- **Code comparisons**: Show the old/bad way vs the new/good way
- **Common pitfalls**: What mistakes do people make?
- **Real-world context**: Where is this used in actual models/systems?
- **Concrete examples**: Small, runnable snippets that demonstrate concepts

Avoid:
- Generic fluff ("This is a powerful technique...")
- Assuming too much knowledge
- Walls of text without code
- Overexplaining trivial concepts

### Step 3.5: Including Plots and Visualizations

For workbooks where visual output is essential (matplotlib, plotting libraries, architecture diagrams):

**Directory Structure:**
```
assets/workbooks/<namespace>/
├── index.html
└── images/
    ├── scatter_example.png
    ├── heatmap_example.png
    └── subplot_example.png
```

**When to Include Visuals:**
- Plot-focused workbooks (matplotlib, seaborn, plotly)
- Architecture diagrams (transformer blocks, network layers)
- Visual comparisons (before/after transformations)
- Examples where output is inherently visual

**Manifest Integration:**
```toml
[[exercises]]
slug = "scatter-plot"
title = "Create Basic Scatter Plot"
kata = "matplotlib_scatter"
objective = "Plot x-y data with markers and labels."
assets = ["images/scatter_example.png"]  # Relative to workbook directory
```

**HTML Embedding:**
```html
<div class="visual-reference">
  <img src="images/scatter_example.png"
       alt="Example scatter plot with labeled axes"
       style="max-width: 100%; border-radius: 8px; margin: 16px 0;">
  <p class="caption">Expected output: scatter plot with proper axis labels</p>
</div>
```

**Workflow:**
1. Create a script to generate reference plots (e.g., `generate_plots.py`)
2. Save images to `assets/workbooks/<namespace>/images/`
3. Use descriptive filenames matching exercise slugs
4. Reference in manifest `assets` field
5. Embed in HTML template with `<img>` tags
6. Include alt text for accessibility
7. Add captions explaining what to observe

**Image Guidelines:**
- **Format**: PNG for plots (crisp text), SVG for diagrams (scalable)
- **Size**: Keep under 200KB per image, resize if needed
- **Dimensions**: Max 800px wide (responsive design)
- **Naming**: `<exercise_slug>_example.png` or `<concept>_diagram.png`
- **Purpose**: Show expected output, not decorative
- **Captions**: Explain what makes the plot correct/good

**Best Practices:**
- Only include images that clarify concepts
- Don't rely solely on images (text should stand alone)
- Show minimal, focused examples (not busy/cluttered plots)
- For comparisons, use side-by-side layout
- Keep visual style consistent across workbook
- Optimize file sizes (compress PNGs)

### Step 4: Design Exercise Progression

Order exercises so that:
1. Each builds on previous skills
2. Complexity increases gradually
3. Dependencies are respected (use `dependencies` field)
4. Early exercises establish fundamentals
5. Later exercises combine multiple concepts

Each exercise card should:
- Explain **why** this operation matters (not just what it does)
- List concrete acceptance criteria
- Provide hints that guide thinking without spoiling solutions
- Show the full function signature and template

## Design Guidelines

### Visual Consistency

Use the provided template without modifying the core aesthetic:
- Typography: Crimson Pro (headings), DM Sans (body), Fira Code (code)
- Colors: Cream/paper backgrounds with rust/navy/sage accents
- Layout: Card-based with subtle shadows and staggered animations

### Customization Points

You can customize:
- Content structure (add/remove sections as needed)
- Number of exercises
- Code comparison grids (1-column for mobile-friendly)
- Additional callout boxes using `.intro-block` style

Don't customize:
- Core color palette (maintain consistency across workbooks)
- Typography choices
- Card hover effects
- Animation timing

### Content Style

**Voice**: Direct, technical, no fluff
- "Turn BCHW images into non-overlapping flattened patches."
- Not: "In this exciting exercise, we'll explore how to transform images..."

**Code**: Show, don't tell
- Prefer side-by-side comparisons over verbal explanations
- Include function signatures with type hints
- Use comments sparingly (only for non-obvious logic)

**Complexity**: Assume competence but explain context
- Don't explain what `torch.Tensor` is
- Do explain why you'd want to split attention heads
- Link to external resources for deep dives

## Workbook Types

### Arenas

Focused practice environments for building muscle memory:
- **Target**: Specific library or operation family (einops, backprop, etc.)
- **Goal**: Fluency and automaticity
- **Structure**: 4-8 exercises, progressive complexity
- **Example**: Einops Arena

### Foundations

Prerequisite knowledge for advanced topics:
- **Target**: Fundamental concepts (attention, convolutions, loss functions)
- **Goal**: Solid conceptual understanding
- **Structure**: 5-10 exercises, building blocks first
- **Example**: (Future) Attention Mechanisms Foundation

### Implementations

End-to-end implementations of papers or architectures:
- **Target**: Full model or algorithm
- **Goal**: Understand how pieces fit together
- **Structure**: 8-15 exercises, following architecture flow
- **Example**: (Future) GPT-2 Implementation

### Deep Dives

Thorough exploration of specific advanced topics:
- **Target**: Complex mechanism or technique
- **Goal**: Mastery and intuition
- **Structure**: 6-12 exercises, theory + practice
- **Example**: (Future) FSRS Spaced Repetition Deep Dive

## Integration with TUI

Workbooks are browsable in the TUI:
1. Press `w` to open Workbooks view
2. Select a workbook to see its exercises
3. Press `a` to add an exercise to your practice queue
4. Press `p` to practice immediately

The TUI reads from `workbooks/*/manifest.toml` to discover workbooks.

## Best Practices

### Kata Selection

- **Atomic focus**: Each kata should teach one concept
- **Real-world relevance**: Operations you'll actually use
- **Progressive difficulty**: Start with building blocks
- **Reusable skills**: Prefer general patterns over one-off tricks

### Exercise Ordering

Bad progression:
```
1. Simple operation
2. Complex multi-step operation  ← Too big a jump
3. Another simple operation      ← Should come earlier
```

Good progression:
```
1. Basic reshape
2. Reshape with grouping
3. Reshape + reduction
4. Complex multi-step operation using previous skills
```

### Content Density

- Introduction: 2-4 paragraphs + code comparison
- Each exercise: 1-2 sentence objective + criteria + hints + template
- Resources: 3-5 high-quality links (official docs > blog posts)

Too sparse: Just listing exercises without context
Too dense: Multi-paragraph essays for each exercise

### Code Examples

Good:
```python
# Before: cryptic and error-prone
x = x.reshape(b, -1, h, d).transpose(1, 2)

# After: explicit and readable
x = rearrange(x, 'b n (h d) -> b h n d', h=num_heads)
```

Bad:
```python
# This code demonstrates the use of the rearrange function
# from the einops library to perform a reshape operation
# that splits the hidden dimension...
x = rearrange(x, 'b n (h d) -> b h n d', h=num_heads)
```

## Maintenance

### When to Update

- **Kata changes**: If a kata's template or signature changes, update the workbook
- **New resources**: Add links to better documentation as it becomes available
- **User feedback**: If exercises are too hard/easy, adjust hints or reorder
- **Dependencies**: If prerequisites change, update manifest

### Versioning

Workbooks don't have explicit versions. Changes should be:
- **Backward compatible**: Don't break links or change kata IDs
- **Incremental**: Update content gradually, not wholesale rewrites
- **Documented**: Note major changes in git commit messages

## Examples

### Minimal Workbook (3 exercises)

For a focused topic like "Softmax Variants":
1. Basic softmax implementation
2. Numerically stable softmax
3. Softmax with temperature scaling

### Standard Workbook (4-6 exercises)

Most workbooks, like Einops Arena:
1. Foundation operation
2. Build on foundation
3. Combine skills
4. Complex real-world application

### Comprehensive Workbook (8-12 exercises)

For implementing an architecture like GPT-2:
1-3. Core components (attention, MLP, embeddings)
4-6. Combining components (blocks, stacking)
7-9. Training infrastructure (loss, optimization)
10-12. Generation and sampling

## Future Enhancements

Potential additions to the workbook system:
- Interactive visualizations (d3.js, if truly helpful for understanding)
- Embedded test runners (run tests in browser)
- Progress tracking (integration with TUI database)
- Difficulty ratings per exercise
- Estimated time to complete
- Community contributions (user-submitted workbooks)

## Questions?

If you're creating a workbook and get stuck:
1. Look at `assets/workbooks/einops_arena/index.html` as a reference
2. Check the template documentation in `assets/workbooks/_template/README.md`
3. Review existing manifests in `workbooks/*/manifest.toml`
4. Keep it simple - less is more for educational content
