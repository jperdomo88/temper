"""
Render TAO_Working_Paper.md to a clean PDF using markdown + WeasyPrint.

Target: ~10-12 page academic-style PDF. Serif body, sans-serif headers,
moderate margins, footer pagination. Tables and code blocks styled for
readability.
"""

from pathlib import Path
import markdown
from weasyprint import HTML, CSS

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
SRC = HERE / "TAO_Working_Paper.md"
OUT = REPO_ROOT / "TAO_Working_Paper.pdf"  # PDF lives at repo root for visibility

md_text = SRC.read_text()

# Strip the H1 (we render it ourselves in the cover block)
lines = md_text.split("\n")
title_block_end = 0
for i, line in enumerate(lines):
    if line.strip() == "---" and i > 5:
        title_block_end = i
        break

# Extract paper title and metadata for cover
title = "Detecting Semantic Laundering in Agentic AI"
subtitle = "A Working Paper on TAO"
author = "Jorge Perdomo"
email = "jorgeperdom@gmail.com"
version = "0.12 · 2026-05-17 · Working draft"

# Body is everything after the first "---"
body_md = "\n".join(lines[title_block_end + 1:])

html_body = markdown.markdown(
    body_md,
    extensions=["tables", "fenced_code", "footnotes", "smarty"],
    output_format="html5",
)

cover_html = f"""
<div class="cover">
  <div class="cover-title">{title}</div>
  <div class="cover-subtitle">{subtitle}</div>
  <div class="cover-author">{author}</div>
  <div class="cover-email">{email}</div>
  <div class="cover-version">{version}</div>
</div>
"""

html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
</head>
<body>
  {cover_html}
  <div class="body-content">
  {html_body}
  </div>
</body>
</html>
"""

CSS_TEXT = """
@page {
  size: letter;
  margin: 0.95in 0.85in 0.95in 0.85in;
  @bottom-center {
    content: counter(page);
    font-family: 'DejaVu Serif', Georgia, serif;
    font-size: 9pt;
    color: #555;
  }
}

@page :first {
  @bottom-center { content: ""; }
}

html {
  font-size: 11pt;
}

body {
  font-family: 'DejaVu Serif', Georgia, 'Liberation Serif', serif;
  line-height: 1.42;
  color: #1a1a1a;
  -weasy-hyphens: auto;
}

.cover {
  text-align: left;
  margin-top: 2.5in;
  padding-bottom: 1.5in;
  page-break-after: always;
}

.cover-title {
  font-family: 'DejaVu Sans', Helvetica, Arial, sans-serif;
  font-size: 26pt;
  font-weight: 700;
  line-height: 1.15;
  color: #111;
  margin-bottom: 0.3em;
}

.cover-subtitle {
  font-family: 'DejaVu Sans', Helvetica, Arial, sans-serif;
  font-size: 16pt;
  font-weight: 400;
  color: #444;
  margin-bottom: 1.8em;
}

.cover-author {
  font-size: 12pt;
  font-weight: 600;
  color: #222;
  margin-top: 0.3em;
}

.cover-email {
  font-size: 10.5pt;
  font-style: italic;
  color: #555;
  margin-top: 0.1em;
  margin-bottom: 1.4em;
}

.cover-version {
  font-size: 10pt;
  color: #666;
  margin-top: 0.2em;
}

h1 {
  font-family: 'DejaVu Sans', Helvetica, Arial, sans-serif;
  font-size: 16pt;
  font-weight: 700;
  color: #111;
  margin-top: 1.2em;
  margin-bottom: 0.4em;
  page-break-after: avoid;
  border-bottom: 1px solid #ccc;
  padding-bottom: 0.2em;
}

h2 {
  font-family: 'DejaVu Sans', Helvetica, Arial, sans-serif;
  font-size: 13pt;
  font-weight: 700;
  color: #111;
  margin-top: 1.4em;
  margin-bottom: 0.35em;
  page-break-after: avoid;
}

h3 {
  font-family: 'DejaVu Sans', Helvetica, Arial, sans-serif;
  font-size: 11.5pt;
  font-weight: 700;
  color: #222;
  margin-top: 1.0em;
  margin-bottom: 0.25em;
  page-break-after: avoid;
}

p {
  margin: 0 0 0.7em 0;
  text-align: justify;
  orphans: 3;
  widows: 3;
}

strong {
  font-weight: 700;
  color: #111;
}

em {
  font-style: italic;
}

ul, ol {
  margin: 0 0 0.8em 0;
  padding-left: 1.5em;
}

li {
  margin-bottom: 0.3em;
  text-align: justify;
}

table {
  border-collapse: collapse;
  margin: 0.8em 0;
  font-size: 10pt;
  width: 100%;
}

th, td {
  border: 1px solid #ccc;
  padding: 5px 8px;
  text-align: left;
  vertical-align: top;
}

th {
  background: #f0f0f0;
  font-family: 'DejaVu Sans', Helvetica, sans-serif;
  font-weight: 700;
}

code {
  font-family: 'DejaVu Sans Mono', 'Liberation Mono', monospace;
  font-size: 9.5pt;
  background: #f4f4f4;
  padding: 1px 3px;
  border-radius: 2px;
}

pre {
  background: #f7f7f7;
  padding: 8px 10px;
  border-left: 3px solid #999;
  font-size: 9pt;
  line-height: 1.3;
  overflow-x: auto;
}

hr {
  border: 0;
  border-top: 1px solid #ccc;
  margin: 1.4em 0;
}

a {
  color: #1f5fa8;
  text-decoration: none;
}

.body-content > p:first-of-type strong:first-child {
  /* Abstract label */
}
"""

HTML(string=html_doc).write_pdf(OUT, stylesheets=[CSS(string=CSS_TEXT)])
print(f"Wrote {OUT}")
