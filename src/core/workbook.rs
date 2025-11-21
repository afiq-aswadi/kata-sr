//! Workbook loading and HTML generation.
//!
//! Workbooks group related katas into a guided learning path. Each workbook
//! is defined by a manifest at `workbooks/<topic>/manifest.toml` and renders
//! to `assets/workbooks/<topic>/index.html`.

use crate::core::kata_loader;
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Top-level workbook metadata from manifest.
#[derive(Debug, Clone, Deserialize)]
pub struct WorkbookMeta {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub learning_goals: Vec<String>,
    #[serde(default)]
    pub prerequisites: Vec<String>,
    #[serde(default)]
    pub resources: Vec<WorkbookResource>,
    #[serde(default)]
    pub kata_namespace: Option<String>,
}

/// External learning resources.
#[derive(Debug, Clone, Deserialize)]
pub struct WorkbookResource {
    pub title: String,
    pub url: String,
}

/// Exercise entry in a workbook.
#[derive(Debug, Clone, Deserialize)]
pub struct WorkbookExercise {
    pub slug: String,
    pub title: String,
    pub kata: String,
    pub objective: String,
    #[serde(default)]
    pub acceptance: Vec<String>,
    #[serde(default)]
    pub hints: Vec<String>,
    #[serde(default)]
    pub assets: Vec<String>,
    #[serde(default)]
    pub dependencies: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct WorkbookManifest {
    workbook: WorkbookMeta,
    exercises: Vec<WorkbookExercise>,
}

/// In-memory representation of a loaded workbook.
#[derive(Debug, Clone)]
pub struct Workbook {
    pub meta: WorkbookMeta,
    pub exercises: Vec<WorkbookExercise>,
    pub manifest_path: PathBuf,
    pub html_path: PathBuf,
}

/// Load all workbooks from `workbooks/<topic>/manifest.toml`.
pub fn load_workbooks() -> Result<Vec<Workbook>> {
    let workbooks_dir = Path::new("workbooks");
    if !workbooks_dir.exists() {
        return Ok(Vec::new());
    }

    let available_katas = kata_loader::load_available_katas()?;
    let katas_by_name: HashMap<String, kata_loader::AvailableKata> = available_katas
        .into_iter()
        .map(|k| (k.name.clone(), k))
        .collect();

    let mut workbooks = Vec::new();
    let mut seen_ids = HashSet::new();

    for entry in fs::read_dir(workbooks_dir).context("failed to read workbooks directory")? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let manifest_path = path.join("manifest.toml");
        if !manifest_path.exists() {
            continue;
        }

        let manifest = load_manifest(&manifest_path)?;
        validate_workbook(&manifest, &katas_by_name)?;

        if !seen_ids.insert(manifest.workbook.id.clone()) {
            bail!("duplicate workbook id detected: {}", manifest.workbook.id);
        }

        let html_path = Path::new("assets")
            .join("workbooks")
            .join(&manifest.workbook.id)
            .join("index.html");

        workbooks.push(Workbook {
            meta: manifest.workbook,
            exercises: manifest.exercises,
            manifest_path: manifest_path.clone(),
            html_path,
        });
    }

    Ok(workbooks)
}

fn load_manifest(manifest_path: &Path) -> Result<WorkbookManifest> {
    let content = fs::read_to_string(manifest_path)
        .with_context(|| format!("failed to read manifest at {}", manifest_path.display()))?;
    let manifest: WorkbookManifest = toml::from_str(&content).with_context(|| {
        format!(
            "failed to parse workbook manifest at {}",
            manifest_path.display()
        )
    })?;
    Ok(manifest)
}

fn validate_workbook(
    manifest: &WorkbookManifest,
    katas_by_name: &HashMap<String, kata_loader::AvailableKata>,
) -> Result<()> {
    if manifest.workbook.id.trim().is_empty() {
        bail!("workbook id cannot be empty");
    }
    if manifest.workbook.title.trim().is_empty() {
        bail!("workbook title cannot be empty");
    }
    if manifest.exercises.is_empty() {
        bail!("workbook must contain at least one exercise");
    }

    let mut seen_slugs = HashSet::new();
    for exercise in &manifest.exercises {
        if exercise.slug.trim().is_empty() {
            bail!("exercise slug cannot be empty");
        }
        if !seen_slugs.insert(exercise.slug.clone()) {
            bail!("duplicate exercise slug detected: {}", exercise.slug);
        }
        if exercise.title.trim().is_empty() {
            bail!("exercise title cannot be empty");
        }
        if exercise.objective.trim().is_empty() {
            bail!("exercise objective cannot be empty");
        }
        if !katas_by_name.contains_key(&exercise.kata) {
            bail!(
                "kata '{}' referenced in exercise '{}' is missing",
                exercise.kata,
                exercise.slug
            );
        }
    }

    for exercise in &manifest.exercises {
        for dep in &exercise.dependencies {
            if !seen_slugs.contains(dep) {
                bail!(
                    "exercise '{}' depends on unknown slug '{}'",
                    exercise.slug,
                    dep
                );
            }
        }
    }

    Ok(())
}

/// Generate HTML for a workbook at its configured output path.
pub fn generate_workbook_html(workbook: &Workbook) -> Result<()> {
    let snippets: Vec<Option<String>> = workbook
        .exercises
        .iter()
        .map(|ex| load_template_snippet(&ex.kata).ok())
        .collect();

    let html = render_html(workbook, &snippets);
    if let Some(parent) = workbook.html_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(&workbook.html_path, html)
        .with_context(|| format!("failed to write {}", workbook.html_path.display()))?;
    Ok(())
}

fn render_html(workbook: &Workbook, snippets: &[Option<String>]) -> String {
    let goals = workbook
        .meta
        .learning_goals
        .iter()
        .map(|g| format!("<li>{}</li>", escape_html(g)))
        .collect::<Vec<_>>()
        .join("");

    let prerequisites = workbook
        .meta
        .prerequisites
        .iter()
        .map(|p| format!("<span class=\"pill\">{}</span>", escape_html(p)))
        .collect::<Vec<_>>()
        .join("");

    let resources = workbook
        .meta
        .resources
        .iter()
        .map(|r| {
            format!(
                "<li><a href=\"{url}\" target=\"_blank\" rel=\"noreferrer noopener\">{title}</a></li>",
                url = escape_html(&r.url),
                title = escape_html(&r.title),
            )
        })
        .collect::<Vec<_>>()
        .join("");

    let exercises = workbook
        .exercises
        .iter()
        .enumerate()
        .map(|(idx, ex)| render_exercise(idx, ex, snippets.get(idx).and_then(|s| s.clone())))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title} · Arena Workbook</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --border: #1f2937;
      --accent: #22d3ee;
      --muted: #94a3b8;
      --text: #e2e8f0;
      --pill: #1d4ed8;
      --pill-text: #e0f2fe;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Inter", "SF Pro Display", "Segoe UI", sans-serif;
      background: radial-gradient(120% 80% at 10% 10%, rgba(34,211,238,0.12), rgba(34,211,238,0) 55%), var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding: 32px 24px 64px;
    }}
    .page {{
      max-width: 1100px;
      margin: 0 auto;
    }}
    h1 {{
      font-size: 32px;
      margin: 0 0 8px 0;
      letter-spacing: -0.02em;
    }}
    .summary {{
      color: var(--muted);
      max-width: 780px;
      margin-bottom: 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 320px;
      gap: 20px;
    }}
    @media (max-width: 1024px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
    .panel {{
      background: linear-gradient(145deg, rgba(17,24,39,0.9), rgba(15,23,42,0.92));
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px 18px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    }}
    .section-title {{
      font-size: 14px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    ul.goals {{
      list-style: disc;
      padding-left: 18px;
      margin: 0;
      color: var(--text);
    }}
    .pill {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--pill);
      color: var(--pill-text);
      font-size: 12px;
      margin: 4px 6px 4px 0;
    }}
    .resources ul {{
      padding-left: 16px;
      margin: 0;
    }}
    .resources a {{
      color: var(--accent);
      text-decoration: none;
    }}
    .resources a:hover {{ text-decoration: underline; }}
    .exercise {{
      margin-bottom: 14px;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 16px;
      background: linear-gradient(145deg, rgba(30,41,59,0.75), rgba(15,23,42,0.9));
    }}
    .exercise h3 {{
      margin: 0 0 6px 0;
      font-size: 18px;
      letter-spacing: -0.01em;
    }}
    .ex-meta {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .objective {{
      margin: 8px 0;
    }}
    pre {{
      background: #0b1223;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      overflow-x: auto;
      font-size: 13px;
      line-height: 1.5;
      color: #cbd5e1;
      margin-top: 10px;
    }}
    .list {{
      padding-left: 16px;
      margin: 6px 0;
      color: var(--text);
    }}
    .list li {{ margin: 2px 0; }}
    .badges {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 8px;
    }}
    .note {{ color: var(--muted); font-size: 13px; margin-top: 8px; }}
  </style>
</head>
<body>
  <main class="page">
    <header>
      <h1>{title}</h1>
      <p class="summary">{summary}</p>
    </header>
    <section class="grid">
      <div>
        {exercises}
      </div>
      <div class="panel">
        <div class="section-title">Learning goals</div>
        <ul class="goals">{goals}</ul>
        <div class="section-title" style="margin-top:14px;">Prerequisites</div>
        <div>{prereq}</div>
        <div class="section-title" style="margin-top:14px;">Resources</div>
        <div class="resources">
          <ul>{resources}</ul>
        </div>
      </div>
    </section>
  </main>
</body>
</html>
"#,
        title = escape_html(&workbook.meta.title),
        summary = escape_html(&workbook.meta.summary),
        goals = goals,
        prereq = prerequisites,
        resources = resources,
        exercises = exercises,
    )
}

fn render_exercise(idx: usize, ex: &WorkbookExercise, snippet: Option<String>) -> String {
    let acceptance = bullet_list(&ex.acceptance, "Acceptance");
    let hints = bullet_list(&ex.hints, "Hints");
    let assets = bullet_list(&ex.assets, "Assets");
    let deps = if ex.dependencies.is_empty() {
        String::new()
    } else {
        let pills = ex
            .dependencies
            .iter()
            .map(|d| format!("<span class=\"pill\">{}</span>", escape_html(d)))
            .collect::<Vec<_>>()
            .join("");
        format!("<div class=\"badges\">{}</div>", pills)
    };

    format!(
        r#"<article class="exercise">
  <div class="ex-meta">Exercise {num} · {slug} · Kata name: {kata}</div>
  <h3>{title}</h3>
  <p class="objective">{objective}</p>
  {acceptance}
  {hints}
  {assets}
  {deps}
  <div class="note">Add later via TUI: Library → All Katas → "{kata}" (or press w for Workbooks, then a/p).</div>
  {snippet}
</article>"#,
        num = idx + 1,
        slug = escape_html(&ex.slug),
        kata = escape_html(&ex.kata),
        title = escape_html(&ex.title),
        objective = escape_html(&ex.objective),
        acceptance = acceptance,
        hints = hints,
        assets = assets,
        deps = deps,
        snippet = snippet
            .map(|s| format!("<pre><code>{}</code></pre>", escape_html(&s)))
            .unwrap_or_default(),
    )
}

fn bullet_list(items: &[String], title: &str) -> String {
    if items.is_empty() {
        return String::new();
    }
    let list = items
        .iter()
        .map(|i| format!("<li>{}</li>", escape_html(i)))
        .collect::<Vec<_>>()
        .join("");
    format!(
        r#"<div>
  <div class="section-title" style="margin-bottom:4px;">{}</div>
  <ul class="list">{}</ul>
</div>"#,
        escape_html(title),
        list
    )
}

fn escape_html(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn load_template_snippet(kata_name: &str) -> Result<String> {
    let katas_root = std::env::var("KATA_SR_KATAS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("katas"));
    let template_path = katas_root
        .join("exercises")
        .join(kata_name)
        .join("template.py");

    let content = fs::read_to_string(&template_path).with_context(|| {
        format!(
            "failed to read template for kata '{}' at {}",
            kata_name,
            template_path.display()
        )
    })?;

    // Keep the snippet short to fit the page; truncate after ~80 lines.
    let snippet: String = content
        .lines()
        .take(80)
        .map(|l| l.to_string())
        .collect::<Vec<_>>()
        .join("\n");

    Ok(snippet)
}
