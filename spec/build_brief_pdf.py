"""
Render TAO_Plain_Language_Brief.md to a clean 2-page PDF (WeasyPrint).

Warm, readable, non-academic styling — this is the doorway document a
non-technical reader sees before the working paper. No cover page.
"""

from pathlib import Path
import markdown
from weasyprint import HTML, CSS

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
SRC = HERE / "TAO_Plain_Language_Brief.md"
OUT = REPO_ROOT / "TAO_Plain_Language_Brief.pdf"

md_text = SRC.read_text()
html_body = markdown.markdown(
    md_text, extensions=["tables", "fenced_code", "smarty"], output_format="html5"
)

html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" /><title>TAO — The Short Version</title></head>
<body><div class="body-content">{html_body}</div></body></html>"""

CSS_TEXT = """
@page {
  size: letter;
  margin: 0.6in 0.8in 0.55in 0.8in;
  @bottom-center { content: counter(page); font-family: 'DejaVu Sans', sans-serif; font-size: 8pt; color: #999; }
}
html { font-size: 9.9pt; }
body {
  font-family: 'DejaVu Serif', Georgia, serif;
  line-height: 1.32; color: #1c1c1c; -weasy-hyphens: auto;
}
h1 {
  font-family: 'DejaVu Sans', Helvetica, Arial, sans-serif;
  font-size: 19pt; font-weight: 700; color: #111;
  margin: 0 0 0.1em 0; line-height: 1.1;
}
/* the italic tagline + byline right after H1 */
h1 + p { color: #555; font-style: italic; margin: 0 0 0.15em 0; font-size: 9.3pt; }
h1 + p + p { color: #333; margin: 0 0 0.3em 0; font-weight: bold; font-style: normal; }
h2 {
  font-family: 'DejaVu Sans', Helvetica, Arial, sans-serif;
  font-size: 11.5pt; font-weight: 700; color: #1a3a5c;
  margin: 0.85em 0 0.28em 0; padding-bottom: 0.1em;
  border-bottom: 1.5px solid #d8e0e8;
}
p { margin: 0 0 0.5em 0; }
strong { color: #111; }
ul { margin: 0.25em 0 0.55em 0; padding-left: 1.1em; }
li { margin-bottom: 0.3em; }
hr { border: none; border-top: 1px solid #ccc; margin: 0.6em 0; }
hr + p, em { color: #444; }
a { color: #1a4a7a; text-decoration: none; }
/* keep headings with their following text */
h2 { -weasy-break-after: avoid; }
"""

HTML(string=html_doc).write_pdf(OUT, stylesheets=[CSS(string=CSS_TEXT)])
print(f"wrote {OUT}")
