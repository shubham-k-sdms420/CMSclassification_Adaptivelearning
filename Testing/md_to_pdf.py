#!/usr/bin/env python3
"""Convert sgd_learning_metrics.md to PDF using markdown + weasyprint."""
import sys
from pathlib import Path

def main():
    base = Path(__file__).resolve().parent
    md_path = base / "sgd_learning_metrics.md"
    pdf_path = base / "sgd_learning_metrics.pdf"

    if not md_path.exists():
        print(f"Error: {md_path} not found")
        sys.exit(1)

    try:
        import markdown
    except ImportError:
        print("Install: pip install markdown")
        sys.exit(1)
    try:
        from weasyprint import HTML, CSS
    except ImportError:
        print("Install: pip install weasyprint")
        sys.exit(1)

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "nl2br"],
        extension_configs={"tables": {}},
    )

    style = """
    @page { size: A4; margin: 2cm; }
    body { font-family: DejaVu Sans, sans-serif; font-size: 10pt; line-height: 1.4; color: #333; }
    h1 { font-size: 18pt; margin-top: 0; border-bottom: 2px solid #333; padding-bottom: 4px; }
    h2 { font-size: 14pt; margin-top: 16px; border-bottom: 1px solid #666; }
    h3 { font-size: 12pt; margin-top: 12px; }
    h4 { font-size: 11pt; margin-top: 8px; }
    table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9pt; }
    th, td { border: 1px solid #999; padding: 4px 6px; text-align: left; }
    th { background: #eee; font-weight: bold; }
    tr:nth-child(even) { background: #f9f9f9; }
    ul, ol { margin: 4px 0; padding-left: 24px; }
    li { margin: 2px 0; }
    strong { font-weight: bold; }
    code { background: #f0f0f0; padding: 1px 4px; font-size: 9pt; }
    hr { border: none; border-top: 1px solid #ccc; margin: 12px 0; }
    p { margin: 6px 0; }
    """

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>SGD Learning Metrics Report</title>
</head>
<body>
{html_body}
</body>
</html>
"""

    html_file = base / "sgd_learning_metrics_temp.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_doc)

    HTML(string=html_doc).write_pdf(
        pdf_path,
        stylesheets=[CSS(string=style)],
    )
    html_file.unlink(missing_ok=True)

    print(f"Created: {pdf_path}")


if __name__ == "__main__":
    main()
