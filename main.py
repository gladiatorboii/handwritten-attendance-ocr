import time
program_start = time.perf_counter()
import os
import sys
from pathlib import Path
from tkinter import Tk, filedialog

#uvicorn api:app --host 0.0.0.0 --port 8000
#http://127.0.0.1:8000/docs
#http://192.168.68.108:8000/docs

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent

from pipeline import AttendancePipeline
from pdf_processor import process_pdf

# A hidden root window -- filedialog needs one to exist, but nothing
# about this tool has an actual window of its own to show.
_tk_root = Tk()
_tk_root.withdraw()

INPUT_FILE = filedialog.askopenfilename(
    title="Select an attendance register to process",
    initialdir=os.path.join(PROJECT_ROOT, "inputs"),
    filetypes=[
        ("Attendance registers", "*.pdf *.jpg *.jpeg *.png"),
        ("PDF files", "*.pdf"),
        ("Image files", "*.jpg *.jpeg *.png"),
        ("All files", "*.*"),
    ],
)

_tk_root.destroy()

if not INPUT_FILE:
    print("No file selected -- exiting.")
    sys.exit(0)

pipeline = AttendancePipeline()

extension = Path(INPUT_FILE).suffix.lower()

if extension in [".jpg", ".jpeg", ".png"]:

    result = pipeline.run(INPUT_FILE)

    print(f"Extracted {result['total_records']} records")
    print("JSON saved successfully.")

elif extension == ".pdf":

    print("PDF detected.")

    process_pdf(pipeline, INPUT_FILE)

else:

    raise ValueError(f"Unsupported file type: {extension}")

print()
print("=" * 60)
print(f"TOTAL PROGRAM TIME : {time.perf_counter() - program_start:.2f} sec")
print("=" * 60)