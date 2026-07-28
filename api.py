"""
REST API wrapper around the attendance OCR pipeline, for deployment on a
company server (main.py's tkinter file picker is a local-desktop-only
entry point, not something a server can run).

Wraps AttendancePipeline / process_pdf exactly as they already exist --
no extraction/validation logic is duplicated here, only HTTP plumbing:
accept an uploaded file, save it to a request-scoped temp location
(so concurrent requests can never collide on the same filename or the
same preprocessed/output paths), run it through the same pipeline
main.py uses, return the same JSON structure.

POST /extract returns immediately with a job_id and status
"processing" (a PDF can take 100-200+ seconds, too long to hold an
HTTP connection open for) -- GET /extract/{job_id} is then polled until
status flips to "completed" or "failed".

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000

Interactive API docs (Swagger UI) are then available at /docs.
"""
import base64
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.middleware.base import BaseHTTPMiddleware

from pipeline import AttendancePipeline
from pdf_processor import process_pdf
from config import get_output_paths

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}


class ResponseTimeMiddleware(BaseHTTPMiddleware):
    """
    Stamps every response with an X-Process-Time header (seconds, wall
    clock) measuring how long this server took to handle that specific
    request -- visible in Swagger UI's response panel after "Execute"
    for any endpoint, not just /extract, without digging through server
    logs. Purely observational: never alters the response body or status.
    """

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.4f}"
        return response


app = FastAPI(
    title="Attendance OCR Pipeline API",
    description="Upload a scanned/photographed attendance register (PDF or image) and get back structured, validated attendance records.",
    version="1.0.0",
)
app.add_middleware(ResponseTimeMiddleware)

# Built once at process startup, not per-request -- the constructor
# validates MISTRAL_API_KEY/GROQ_API_KEY are set and raises immediately
# if not, so a misconfigured server fails fast on startup rather than on
# the first upload.
pipeline = AttendancePipeline()


@app.get("/health")
def health():
    return {"status": "ok"}


# ------------------------------------------------------------------
# JOB STORE -- a PDF can take 100-200+ seconds (Mistral OCR + Llama
# recheck per page). Holding an HTTP connection open that long is bad
# practice (proxies/browsers can time it out, and the caller has no
# visibility into progress) -- so POST /extract submits and returns
# immediately with "processing", and the caller polls
# GET /extract/{job_id} until it flips to "completed" (or "failed").
# One upload endpoint, one status-check endpoint -- not two separate
# ways to start the same work.
#
# In-memory dict, guarded by a lock since the background thread and the
# polling requests can touch it at the same time. This only works
# correctly with a single uvicorn worker process (the default -- see
# README) -- a second worker process would have its own separate empty
# dict and could never see a job another worker started. Jobs stay in
# memory for the life of the server process; there's no expiry/cleanup
# yet, so a very long-running deployment would accumulate old job
# records (fine for a demo/moderate-volume server, worth revisiting for
# heavy production use).
# ------------------------------------------------------------------
_jobs = {}
_jobs_lock = threading.Lock()


def _run_job(job_id, saved_path, temp_dir, extension, include_excel):
    def report_progress(current_page, total_pages):
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["progress"] = f"Processing page {current_page} of {total_pages}"

    try:
        if extension == ".pdf":
            result = process_pdf(pipeline, saved_path, progress_callback=report_progress)
        else:
            result = pipeline.run(saved_path, progress_callback=report_progress)

        if include_excel:
            # Binary, unlike the old HTML report -- base64-encoded so it
            # still fits in the same JSON job-status body rather than
            # requiring callers to always use GET /report/{filename}
            # separately just to get the spreadsheet.
            output_json, output_excel = get_output_paths(saved_path)
            if os.path.exists(output_excel):
                with open(output_excel, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("ascii")
                result = {**result, "excel_report_base64": encoded}

        with _jobs_lock:
            _jobs[job_id] = {"status": "completed", "result": result}
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id] = {"status": "failed", "error": str(e)}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/extract", status_code=202)
def extract(background_tasks: BackgroundTasks, file: UploadFile = File(...), include_excel: bool = False):
    """
    Upload one attendance register (PDF, JPG, JPEG, or PNG). Returns
    immediately with a job_id and status "processing" -- the actual
    OCR/recheck/validation work (100-200+ seconds for a PDF) happens
    after this response is sent, not during it.

    Poll GET /extract/{job_id} until status is "completed" (the full
    extracted records are in the "result" field then -- see
    README/DOCUMENTATION.md for its shape) or "failed" (an "error"
    field explains what went wrong). While "processing", a "progress"
    field ("Processing page 3 of 8") gives a sense of how far along it
    is, instead of a flat "processing" for the whole 100-200+ seconds.
    """
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}' -- expected one of {sorted(ALLOWED_EXTENSIONS)}",
        )

    # A UUID-based saved filename (not the caller's original filename)
    # means two concurrent uploads -- even for files with the exact same
    # name -- never collide on the same temp path, preprocessed image
    # path, or outputs/<name>.json path (see config.get_output_paths).
    job_id = uuid.uuid4().hex
    temp_dir = tempfile.mkdtemp(prefix=f"attendance_api_{job_id}_")
    saved_path = os.path.join(temp_dir, f"{job_id}{extension}")

    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    with _jobs_lock:
        _jobs[job_id] = {"status": "processing", "progress": "Starting..."}

    background_tasks.add_task(_run_job, job_id, saved_path, temp_dir, extension, include_excel)

    return {"job_id": job_id, "status": "processing"}


@app.get("/extract/{job_id}")
def get_extract_status(job_id: str):
    """
    Checks on a job started by POST /extract. Returns "processing"
    (with "progress", e.g. "Processing page 3 of 8"), "completed"
    (with "result"), or "failed" (with "error").
    """
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"job_id": job_id, **job}


@app.get("/report/{filename}")
def get_report(filename: str):
    """
    Downloads a previously generated Excel report by output filename
    (the stem returned in a prior /extract call's saved output, e.g. the
    UUID request_id) -- an alternative to include_excel=true when the
    caller wants the report fetched separately from the extraction call.
    """
    output_json, output_excel = get_output_paths(filename)
    if not os.path.exists(output_excel):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(
        output_excel,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=os.path.basename(output_excel),
    )
