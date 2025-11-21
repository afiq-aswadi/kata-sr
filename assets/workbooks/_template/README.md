# Workbook Template

This template provides a consistent design for workbook pages across the kata-sr project.

## Design Aesthetic

- **Typography**: Crimson Pro (serif) for headings, DM Sans for body, Fira Code for code
- **Color Palette**: Warm, paper-like feel with cream backgrounds, rust/navy/sage accents
- **Layout**: Card-based with subtle shadows and animations
- **Theme**: Inspired by scientific journals and lab notebooks

## How to Use

1. Copy `workbook-template.html` to `assets/workbooks/<workbook-name>/index.html`
2. Replace all `{{PLACEHOLDER}}` values with your content:

### Required Placeholders

- `{{WORKBOOK_TITLE}}`: Title of the workbook (e.g., "Einops Arena: Shape Fluency")
- `{{WORKBOOK_TYPE}}`: Category (e.g., "Tensor Operations", "Data Structures")
- `{{WORKBOOK_SUMMARY}}`: One-sentence summary

### Introduction Section

- `{{INTRODUCTION_TEXT}}`: Main introduction paragraph(s)
- `{{BEFORE_CODE}}`: Code showing the problem/old way
- `{{AFTER_CODE}}`: Code showing the solution/new way

### Learning Goals

Repeat the `.goal-item` div for each goal:
```html
<div class="goal-item">
  {{LEARNING_GOAL}}
</div>
```

### Prerequisites

Replace with actual prerequisites:
```html
<span class="badge prereq">{{PREREQ_1}}</span>
<span class="badge prereq">{{PREREQ_2}}</span>
```

### Exercises

Repeat the `.card` article for each exercise, filling in:
- `{{EXERCISE_TITLE}}`: Exercise name
- `{{N}}`: Exercise number
- `{{SLUG}}`: URL-friendly slug
- `{{EXERCISE_OBJECTIVE}}`: Brief description
- `{{ACCEPTANCE_CRITERIA_N}}`: What success looks like
- `{{HINT_N}}`: Helpful hints
- `{{DEPENDENCY}}`: Prerequisites (optional)
- `{{KATA_NAME}}`: Kata ID for TUI
- `{{TEMPLATE_CODE}}`: Code template

### Resources

Repeat list items:
```html
<li><a href="{{URL}}" target="_blank" rel="noopener noreferrer">{{RESOURCE_TITLE}}</a></li>
```

## Customization Tips

- **Colors**: Adjust CSS variables in `:root` for different aesthetics
- **Fonts**: Change Google Fonts imports and font-family declarations
- **Animations**: Modify `@keyframes fadeSlideIn` and animation-delay values
- **Layout**: Adjust `.page` max-width and padding for different spacing

## Sections are Optional

Feel free to remove sections you don't need:
- Introduction can be simplified or removed
- Code comparisons are optional
- Resources can be minimal or extensive

The key is consistency across workbooks while allowing flexibility for content needs.
