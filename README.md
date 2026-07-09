# Attendance Register OCR Pipeline

Extracts daily in/out attendance records (date, in-time, out-time, status, remark) from photographed/scanned handwritten attendance registers — single images or multi-page PDFs.

## How it works

- **Mistral Document AI OCR** reads each page and transcribes it to a markdown table, which is parsed into structured records (date/in_time/out_time/status/remark) by keyword-matching column headers, not fixed column positions — so registers with different column counts, orders, or wording all work.
- **Llama 4 Scout** (via Groq) independently re-reads any row flagged as suspicious (an invalid time format, or a statistical outlier against the rest of the page) as a second opinion, since Mistral re-checking its own read can't catch a misread it makes confidently both times.
- A validation layer flags cells for human review (invalid dates/times, impossible chronology, OCR disagreement between the two engines, struck-through rows, etc.) and produces a confidence score per field.
- Double-page photo spreads (two physical register pages in one shot) are auto-detected and split before OCR ever sees them.

## Setup

1. **Python 3.11** (tested on 3.11.7).
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies (pinned to tested versions):
   ```
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in both API keys:
   ```
   MISTRAL_API_KEY=your_mistral_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   ```
   - Mistral key: https://console.mistral.ai
   - Groq key: https://console.groq.com

## Running

```
python main.py
```

A file picker opens — select any `.pdf`, `.jpg`, `.jpeg`, or `.png` attendance register. The pipeline processes it and writes two output files named after the input file:

- `outputs/<filename>.json` — structured records with full per-cell validation/confidence detail
- `outputs/<filename>.html` — a readable table report, one section per employee/page

Dates and times are saved exactly as OCR'd — nothing is forced/corrected against an assumed month or year.

## Project layout

| File | Purpose |
|---|---|
| `main.py` | Entry point — file picker, dispatches to image or PDF processing |
| `pipeline.py` | Per-page orchestration: crop → deskew → OCR → recheck → validate |
| `pdf_processor.py` | Multi-page PDF driver, calls into `pipeline.py` per page |
| `mistral_ocr_engine.py` | Mistral OCR call + markdown table parsing into records |
| `llama_vision_engine.py` | Llama 4 Scout (Groq) second-opinion recheck on flagged rows |
| `recheck_utils.py` | Engine-agnostic outlier detection / "does this row need a recheck" logic |
| `validation_engine.py` | Per-cell/per-row flagging and confidence scoring |
| `post_processing.py` | Date/time/status string normalization |
| `preprocessing/` | Image crop-to-content and deskew |
| `html_report.py` | Renders the `.html` report |
| `config.py` | API keys (from `.env`), output paths |

## Notes

- `inputs/` holds sample registers used during development/testing.
- `outputs/` is where results land; it's gitignored (contains real attendance data).
- No local model server is required — both OCR and recheck are hosted APIs (Mistral, Groq).
