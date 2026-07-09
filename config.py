import os

from dotenv import load_dotenv

load_dotenv()

# OCR ENGINE
# Mistral's Document AI OCR produces the full-page read (records +
# employee name) -- see mistral_ocr_engine.py for why Surya/EasyOCR were
# removed.
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")

# Row recheck (second opinion on flagged rows) is handled by Llama 4
# Scout via Groq instead of Mistral rechecking itself -- see
# llama_vision_engine.py for why an independent model catches more.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


# DEBUG
DEBUG = False


def debug_print(*args, **kwargs):
    """Print only when DEBUG is enabled, so verbose OCR dumps don't spam
    stdout (and don't risk unicode-encoding crashes) on normal runs."""
    if DEBUG:
        print(*args, **kwargs)


# PATHS
OUTPUT_DIR = "outputs"

PREPROCESSED_DIR = os.path.join(
    OUTPUT_DIR,
    "preprocessed"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)


def get_output_paths(input_path):
    """
    Output filenames mirror whichever input file was actually processed
    (e.g. "inputs/Rahul_June.pdf" -> "outputs/Rahul_June.json" /
    ".html") instead of a single fixed "result.json"/"report.html" that
    every run would overwrite regardless of which file it came from.
    """
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return (
        os.path.join(OUTPUT_DIR, f"{stem}.json"),
        os.path.join(OUTPUT_DIR, f"{stem}.html"),
    )
