import io
import json
import os
import re
import threading
import uuid
import zipfile
import cv2
import numpy as np
import pypdfium2 as pdfium
import pytesseract
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps

load_dotenv()

from db import BillRecord, ChatMessage, engine, init_db
from extract import extract_bill_data
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from rules_engine.calculator import (
    CalcOpts,
    calculate_full,
    load_tariff,
    verify_bill,
)
from sqlmodel import Session, select

app = FastAPI()

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DATA_DIR = os.getenv("DATA_DIR", "media")


class PlanRequest(BaseModel):
    units: int = 0
    present_reading: int | None = None
    previous_reading: int | None = None
    category: str = "auto"  # auto | protected | unprotected | commercial | industrial
    phase: str = "single"  # single | three
    fpa_per_unit: float = 0.0
    qta_amount: float = 0.0
    tv_fee: float = 0.0
    surcharge_percent: float = 0.0
    gst_percent: float | None = None
    electricity_duty_percent: float | None = None


@app.on_event("startup")
def on_startup():
    init_db()


# OCR work is serialized per job; Tesseract is far lighter than EasyOCR/PyTorch
# so this backend fits comfortably on free-tier hosts (~200 MB peak).
_ocr_lock = threading.Lock()
JOBS: dict[str, dict] = {}


OTHER_DISCOS = [
    "LESCO",
    "IESCO",
    "FESCO",
    "PESCO",
    "GEPCO",
    "HESCO",
    "SEPCO",
    "K-ELECTRIC",
    "TESCO",
    "QESCO",
]

MEPCO_ANCHOR_TERMS = ["MEPCO", "MULTAN ELECTRIC", "MULTAN", "MEPCO LTD"]

BILL_ANCHOR_TERMS = [
    "BILL",
    "TARIFF",
    "CONSUMER",
    "REFERENCE",
    "UNITS",
    "METER",
    "READING",
    "DUE DATE",
    "PAYABLE",
    "BREAKDOWN",
    "ELECTRIC",
]


def check_mepco_structure(text_list: list[str]) -> bool:
    """Reject other DISCOs; accept only genuine MEPCO bills.

    Returns True only when:
    - No competing DISCO name is present, AND
    - Either a strong MEPCO marker is present, OR enough generic bill
      anchors exist that this is clearly a Pakistan utility bill.
    """
    full_text = " ".join(text_list).upper()
    if not full_text.strip():
        return False

    for company in OTHER_DISCOS:
        if company in full_text:
            return False

    has_mepco = any(term in full_text for term in MEPCO_ANCHOR_TERMS)
    matching_bill_anchors = sum(1 for term in BILL_ANCHOR_TERMS if term in full_text)

    if has_mepco:
        return True

    return matching_bill_anchors >= 4


def _rotate(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image clockwise by angle degrees (0/90/180/270)."""
    if angle == 0:
        return img.copy()
    code = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }[angle]
    return cv2.rotate(img, code)


def _prepare_gray(
    img: np.ndarray, min_width: int = 3000, max_width: int = 2200
) -> np.ndarray:
    """Upscale-to-readable + grayscale + Otsu binarize for Tesseract.

    Tesseract reads far better from a crisp Otsu-binarized bitmap. Small
    screenshots are upscaled aggressively; large photos are capped so OCR
    stays fast.
    """
    img = img.copy()
    h, w = img.shape[:2]
    if w < min_width:
        scale = min_width / w
    elif w > max_width:
        scale = max_width / w
    else:
        scale = 1.0
    if abs(scale - 1.0) > 1e-6:
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _prepare_probe(img: np.ndarray, max_width: int = 640) -> np.ndarray:
    """Small normalized gray for fast rotation probing."""
    img = img.copy()
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(
            img, (max_width, int(h * (max_width / w))), interpolation=cv2.INTER_AREA
        )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


def _pdf_to_image(contents: bytes, max_pages: int = 2) -> np.ndarray:
    """Extract the sharpest visual of a PDF bill page.

    Scanned/photo bills are PDFs that embed a full-resolution image; grab that
    directly (crisp pixels). If none exists (text-native PDFs), fall back to
    rendering the page via pypdfium2.
    """
    try:
        import pymupdf
    except Exception:
        pymupdf = None

    if pymupdf is not None:
        try:
            doc = pymupdf.open(stream=contents, filetype="pdf")
            candidates = []
            for page_no in range(min(doc.page_count, max_pages)):
                page = doc[page_no]
                for img in page.get_images(full=True):
                    try:
                        info = doc.extract_image(img[0])
                        candidates.append(info)
                    except Exception:
                        continue
            if candidates:
                # Prefer the largest embedded image (most pixels = more detail).
                info = max(candidates, key=lambda i: i.get("width", 0) * i.get("height", 0))
                data = info["image"]
                decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if decoded is not None and decoded.size > 0:
                    return decoded
                # Some embeds are image formats OpenCV can't decode directly.
                pil_img = Image.open(io.BytesIO(data)).convert("RGB")
                return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[PDF WARN] pymupdf image extraction failed, falling back to render: {e}")

    try:
        pdf = pdfium.PdfDocument(contents)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read this PDF ({e}). Please upload a clear JPG, PNG, or WEBP photo of your MEPCO bill.",
        )

    page = pdf[0] if len(pdf) > 0 else None
    if page is None:
        raise HTTPException(status_code=400, detail="PDF appears to be empty.")

    # Render adaptively so even small-page PDFs produce a readable bitmap.
    page_w, page_h = page.get_size()
    long_pt = max(page_w, page_h)
    scale = max(1.5, min(12.0, 1800.0 / max(long_pt, 1)))
    bitmap = page.render(scale=scale)
    pil_img = bitmap.to_pil().convert("RGB")
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _ocr_once(prepared: np.ndarray, temp_path: str) -> tuple[list[str], float]:
    """Run a single Tesseract pass; returns (lines, confidence_score)."""
    cv2.imwrite(temp_path, prepared)
    data = pytesseract.image_to_data(
        temp_path, config="--oem 3 --psm 6", output_type=pytesseract.Output.DICT
    )
    line_text: dict[int, list[str]] = {}
    line_conf: dict[int, list[float]] = {}
    for i, raw in enumerate(data.get("text") or []):
        token = (raw or "").strip()
        if not token:
            continue
        line_no = data["line_num"][i] if i < len(data["line_num"]) else 0
        conf = data["conf"][i] if i < len(data["conf"]) else 0
        line_text.setdefault(line_no, []).append(token)
        line_conf.setdefault(line_no, []).append(float(conf) if conf >= 0 else 0.0)

    lines = [" ".join(line_text[k]) for k in sorted(line_text)]
    score = sum(sum(v) for v in line_conf.values())
    return lines, score


def preprocess_and_ocr(temp_path: str, img: np.ndarray):
    """Auto-stabilize rotation, then OCR.

    1. Apply EXIF orientation so phone photos are upright instantly.
    2. Fast path: OCR once at 0°. If MEPCO validates, return immediately.
    3. Slow path: probe other rotations at low res to find the upright one,
       then re-OCR that orientation at full resolution and validate.
    """
    # --- 1. EXIF orientation correction (near-free) ---
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # --- 2. Fast path: upright, full-res single pass ---
    prepared_0 = _prepare_gray(img)
    lines_0, score_0 = _ocr_once(prepared_0, temp_path)

    if check_mepco_structure(lines_0):
        print("[OK] MEPCO bill detected upright (0 degrees).")
        return lines_0, True, 0

    # --- 3. Probe other rotations at low res to locate upright text ---
    best_angle = 0
    best_score = score_0
    best_lines = lines_0

    for angle in (90, 180, 270):
        rotated = _rotate(img, angle)
        probe = _prepare_probe(rotated, max_width=640)
        probe_lines, probe_score = _ocr_once(probe, temp_path)

        if check_mepco_structure(probe_lines):
            # Found the correct orientation — re-OCR at full quality.
            full = _prepare_gray(_rotate(img, angle))
            lines_full, score_full = _ocr_once(full, temp_path)
            print(f"[OK] MEPCO bill detected at {angle} degrees rotation.")
            return lines_full, True, angle

        if probe_score > best_score:
            best_score = probe_score
            best_angle = angle
            best_lines = probe_lines

    if best_angle != 0:
        full = _prepare_gray(_rotate(img, best_angle))
        lines_full, score_full = _ocr_once(full, temp_path)
        if score_full > best_score:
            best_lines = lines_full
        print(f"[OK] Auto-rotated bill to {best_angle} degrees.")

    return best_lines, check_mepco_structure(best_lines), best_angle


def _run_extraction_job(job_id: str, raw_bytes: bytes, filename: str, is_pdf: bool = False):
    temp_path = f"temp_{job_id}.webp"
    job = JOBS[job_id]
    try:
        job["status"] = "running"
        job["stage"] = "Correcting orientation..."

        if is_pdf:
            img = _pdf_to_image(raw_bytes)
        else:
            np_img = np.frombuffer(raw_bytes, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None or img.size == 0:
            job["status"] = "error"
            job["error"] = "Invalid or unreadable file uploaded."
            return

        job["stage"] = "Reading bill text (OCR)..."
        with _ocr_lock:
            raw_text, is_valid_mepco, used_angle = preprocess_and_ocr(temp_path, img)

        if not is_valid_mepco:
            job["status"] = "error"
            job["error"] = (
                "Invalid Bill Format! Only official MEPCO electricity bills are accepted. "
                "Other DISCO bills or non-bill images are rejected. "
                "Please upload a clear, well-lit MEPCO bill photo."
            )
            return

        job["stage"] = "Extracting bill data with AI..."
        structured_data = extract_bill_data(raw_text)

        if isinstance(structured_data, dict) and structured_data.get("error"):
            job["status"] = "error"
            job["error"] = f"Failed to extract bill data: {structured_data['error']}"
            return

        job["stage"] = "Verifying tariff against NEPRA rules..."
        tariff = load_tariff()
        verification = verify_bill(structured_data, tariff)

        merged = {**structured_data, **verification, "orientation_corrected": used_angle != 0}

        with Session(engine) as session:
            record = BillRecord(
                billing_month=str(structured_data.get("billing_month", "Unknown")),
                units_consumed=float(structured_data.get("units_consumed", 0) or 0),
                total_amount_due=float(structured_data.get("total_amount_due", 0) or 0),
                discrepancy_flag=verification["discrepancy_flag"],
                structured_json=json.dumps(merged),
                image_filename=filename,
                raw_ocr_text="\n".join(raw_text),
                image_path=None,
            )
            session.add(record)
            session.commit()
            record_id = int(record.id)  # assign while the session is still open

            # Persist the original upload so it can be included in the backup.
            try:
                os.makedirs(DATA_DIR, exist_ok=True)
                ext = os.path.splitext(filename)[1].lower() or ".webp"
                dest = os.path.join(DATA_DIR, f"bill_{record_id}{ext}")
                with open(dest, "wb") as f:
                    f.write(raw_bytes)
                record.image_path = dest
                session.add(record)
                session.commit()
            except Exception as media_err:
                print(f"[MEDIA WARN] Could not persist bill image: {media_err}")

        job["status"] = "done"
        job["result"] = {**merged, "id": record_id}
    except Exception as e:
        print(f"[JOB ERROR] {job_id}: {repr(e)}")
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/")
def read_root():
    return {"message": "Bijli Audit backend is running"}


@app.post("/api/v1/calculate-bill")
def calculate_bill(data: PlanRequest):
    tariff = load_tariff()

    # Meter-reading mode: derive units from Present - Previous automatically.
    units = data.units
    if data.present_reading is not None and data.previous_reading is not None:
        if data.present_reading < data.previous_reading:
            raise HTTPException(
                status_code=400,
                detail="Present reading cannot be lower than the previous reading.",
            )
        units = data.present_reading - data.previous_reading
    elif units <= 0:
        units = 0

    result = calculate_full(
        units=units,
        category=data.category,
        phase=data.phase,
        tariff=tariff,
        opts=CalcOpts(
            fpa_per_unit=data.fpa_per_unit,
            qta_amount=data.qta_amount,
            tv_fee=data.tv_fee,
            surcharge_percent=data.surcharge_percent,
            gst_percent=data.gst_percent,
            duty_percent=data.electricity_duty_percent,
        ),
    )
    result["slab_limit"] = max(s["max_units"] for s in tariff.get("protected", []))
    return result


@app.get("/api/v1/tariff")
def get_tariff():
    """Expose the tariff schedule so the UI can render slabs + tooltips."""
    tariff = load_tariff()
    return {
        "disco": tariff["disco"],
        "effective_date": tariff.get("effective_date"),
        "source": tariff.get("source"),
        "fixed_charges": tariff.get("fixed_charges", {}),
        "taxes": tariff.get("taxes", {}),
        "protected_slab_limit": max(
            s["max_units"] for s in tariff.get("protected", [])
        ),
        "categories": {
            "protected": tariff.get("protected", []),
            "unprotected": tariff.get("unprotected", []),
            "commercial": tariff.get("commercial", []),
            "industrial": tariff.get("industrial", []),
        },
    }


@app.post("/extract-bill")
async def extract_bill_async(file: UploadFile = File(...)):
    """Start an async extraction job and return a job id immediately."""
    filename = file.filename or f"bill_{uuid.uuid4().hex[:8]}"

    ext = os.path.splitext(filename)[1].lower()
    is_pdf = ext == ".pdf"
    if ext not in (".pdf", ".jpg", ".jpeg", ".png", ".webp"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a JPG, PNG, WEBP, or PDF of your MEPCO electricity bill.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued", "stage": "Queueing...", "progress": 5, "result": None, "error": None}

    thread = threading.Thread(
        target=_run_extraction_job,
        args=(job_id, contents, filename, is_pdf),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/extract-bill/status/{job_id}")
def extract_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    # Simple progress mapping per stage.
    stage_progress = {
        "Queueing...": 5,
        "Correcting orientation...": 15,
        "Reading bill text (OCR)...": 35,
        "Extracting bill data with AI...": 70,
        "Verifying tariff against NEPRA rules...": 90,
    }
    if job["status"] == "done":
        return {"job_id": job_id, "status": "done", "progress": 100, "stage": "Complete"}
    if job["status"] == "error":
        return {"job_id": job_id, "status": "error", "progress": 100, "stage": "Failed", "error": job.get("error")}

    progress = stage_progress.get(job.get("stage", ""), 20)
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": progress,
        "stage": job.get("stage", ""),
    }


@app.get("/extract-bill/result/{job_id}")
def extract_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] == "error":
        raise HTTPException(status_code=422, detail=job.get("error", "Extraction failed."))
    if job["status"] != "done":
        raise HTTPException(status_code=202, detail="Job still running.")
    return job["result"]


@app.post("/verify-bill")
async def verify(bill_data: dict):
    tariff = load_tariff()
    return verify_bill(bill_data, tariff)


@app.get("/bills")
def list_bills():
    with Session(engine) as session:
        records = session.exec(select(BillRecord).order_by(BillRecord.id)).all()
        return records


@app.get("/bills/{bill_id}")
def get_bill(bill_id: int):
    with Session(engine) as session:
        record = session.get(BillRecord, bill_id)
        if not record:
            raise HTTPException(status_code=404, detail="Bill record not found")
        return record


@app.get("/api/v1/backup")
def download_backup():
    """Bundle every bill, its stored image, and the chat log into one ZIP."""
    with Session(engine) as session:
        records = session.exec(select(BillRecord).order_by(BillRecord.id)).all()
        chats = session.exec(select(ChatMessage).order_by(ChatMessage.id)).all()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        bills_payload = []
        for r in records:
            structured = {}
            try:
                structured = json.loads(r.structured_json or "{}")
            except Exception:
                structured = {}
            bills_payload.append(
                {
                    "id": r.id,
                    "billing_month": r.billing_month,
                    "units_consumed": r.units_consumed,
                    "total_amount_due": r.total_amount_due,
                    "discrepancy_flag": r.discrepancy_flag,
                    "image_filename": r.image_filename,
                    "structured_json": structured,
                    "raw_ocr_text": r.raw_ocr_text,
                    "created_at": str(r.created_at),
                }
            )
            if r.image_path and os.path.exists(r.image_path):
                zf.write(r.image_path, f"bill_images/{os.path.basename(r.image_path)}")

        zf.writestr("bills.json", json.dumps(bills_payload, indent=2))
        zf.writestr(
            "chat_history.json",
            json.dumps(
                [
                    {
                        "id": c.id,
                        "session_id": c.session_id,
                        "role": c.role,
                        "content": c.content,
                        "created_at": str(c.created_at),
                    }
                    for c in chats
                ],
                indent=2,
            ),
        )
        zf.writestr(
            "README.txt",
            "BijliAudit full data backup.\n- bills.json: every audit record with structured JSON + raw OCR text\n- bill_images/: original uploaded bill files\n- chat_history.json: mascot conversation log\n",
        )

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": "attachment; filename=bijli-audit-backup.zip"
        },
    )


@app.get("/chat/history")
def chat_history(session_id: str = "default"):
    with Session(engine) as session:
        rows = session.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id)
        ).all()
        return [
            {"id": c.id, "role": c.role, "content": c.content, "created_at": str(c.created_at)}
            for c in rows
        ]


def _extract_stated_reading(raw_ocr_text: str | None) -> str | None:
    """Hunt for a 'present / current reading' value inside the bill's OCR text."""
    if not raw_ocr_text:
        return None
    lines = [ln.strip() for ln in raw_ocr_text.splitlines() if ln.strip()]
    reading_keywords = ("present reading", "present", "current reading", "read")
    for i, line in enumerate(lines):
        lowered = line.lower()
        if any(k in lowered for k in reading_keywords):
            window = lines[max(0, i - 1): i + 4]
            for candidate in window:
                nums = re.findall(r"\b\d{3,}\b", candidate)
                if nums:
                    return max(nums, key=lambda x: len(x))
    return None


@app.post("/api/v1/meter-check")
async def meter_check(file: UploadFile = File(...), bill_id: int | None = None):
    """Photograph your physical meter; compare the OCR'd reading against a stored bill."""
    filename = file.filename or f"meter_{uuid.uuid4().hex[:8]}"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        raise HTTPException(
            status_code=400,
            detail="Upload a photo of your meter (JPG, PNG, or WEBP).",
        )

    contents = await file.read()
    np_img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if np_img is None:
        raise HTTPException(status_code=400, detail="Could not read the meter photo.")

    temp_path = f"meter_{uuid.uuid4().hex[:8]}.webp"
    try:
        with _ocr_lock:
            prepared = _prepare_gray(np_img)
            lines, _ = _ocr_once(prepared, temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    all_nums: list[str] = []
    for ln in lines:
        all_nums.extend(re.findall(r"\d{3,}", ln))

    # A live meter register is a short run (typically 5-6 digits, rarely more).
    # Filter out reference numbers, phone numbers, dates, and slab values.
    ref_number = None
    try:
        ref_number = str(
            (json.loads(record.structured_json or "{}") if record else {}).get("reference_number", "")
        )
    except Exception:
        ref_number = None

    candidates = []
    for n in all_nums:
        if not (3 <= len(n) <= 8):
            continue
        if ref_number and (ref_number in n or n in ref_number):
            continue
        candidates.append(n)

    reading = None
    if candidates:
        # Longest number-first (the register is the longest run), then largest value.
        reading = max(candidates, key=lambda x: (len(x), int(x)))

    with Session(engine) as session:
        if bill_id is not None:
            record = session.get(BillRecord, bill_id)
        else:
            record = session.exec(select(BillRecord).order_by(BillRecord.id.desc())).first()

    stated = _extract_stated_reading(record.raw_ocr_text if record else None) if record else None

    if reading is None:
        flag = "Inconclusive"
        explanation = "No clear number could be read from the meter photo. Move closer, avoid glare, and try again."
    elif stated is None:
        flag = "Inconclusive"
        explanation = (
            f"Meter read as {reading}, but the bill's 'present reading' field was not legible, "
            "so no automated comparison is possible."
        )
    elif str(reading).strip() != str(stated).strip():
        flag = "Mismatch"
        explanation = f"Meter shows {reading}, but the bill records {stated}. This is worth disputing."
    else:
        flag = "Match"
        explanation = f"Meter shows {reading}, matching the bill's stated reading of {stated}."

    return {
        "meter_reading": reading,
        "stated_reading": stated,
        "bill_id": record.id if record else None,
        "billing_month": record.billing_month if record else None,
        "flag": flag,
        "explanation": explanation,
    }


@app.post("/chat")
async def chat(payload: dict):
    bill_id = payload.get("bill_id")
    session_id = str(payload.get("session_id") or "default")
    user_message = payload.get("message", "").strip()
    if not user_message:
        return {"reply": "Please type a question about your bill."}

    context = payload.get("context")

    if not context and bill_id:
        try:
            with Session(engine) as session:
                record = session.get(BillRecord, int(bill_id))
                if record and record.structured_json:
                    context = record.structured_json
        except Exception as db_err:
            print(f"[DB Error] {db_err}")

    if not context:
        with Session(engine) as session:
            latest = session.exec(
                select(BillRecord).order_by(BillRecord.id.desc())
            ).first()
            if latest and latest.structured_json:
                context = latest.structured_json

    context_block = (
    json.dumps(context, indent=2)
    if isinstance(context, dict)
    else (context if context else "No bill has been uploaded/selected yet.")
)

    # Persist the conversation so follow-up questions stay context-aware.
    try:
        with Session(engine) as session:
            session.add(ChatMessage(session_id=session_id, role="user", content=user_message))
            session.commit()
    except Exception as db_err:
        print(f"[DB WARN] chat history write failed: {db_err}")

    prompt = f"""You are Bijli, a friendly assistant explaining Pakistani MEPCO electricity bills in plain language.
Use ONLY the bill data provided below to answer — never invent numbers, rates, or totals.
If the bill data is empty, tell the user you can't see an uploaded bill yet and invite them to upload one.
Do NOT use Markdown tables, raw pipes (|), or heavy markup. Keep formatting to clean short paragraphs and an occasional bullet list for readability in a small chat widget.

BILL DATA:
{context_block}

USER QUESTION:
{user_message}"""

    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("❌ OPENROUTER_API_KEY is missing or empty in .env!")
            return {"reply": "API key configuration missing. Please check backend .env file."}

        chat_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=90,
        )

        response = chat_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
        )
        reply = (response.choices[0].message.content or "").strip()
        if not reply:
            reply = "I couldn't think of a reply just now. Please try again in a few seconds."
        try:
            with Session(engine) as session:
                session.add(ChatMessage(session_id=session_id, role="assistant", content=reply))
                session.commit()
        except Exception as db_err:
            print(f"[DB WARN] chat history write failed: {db_err}")
        return {"reply": reply}

    except Exception as e:
        print("\n❌ OPENROUTER API ERROR:", str(e), "\n")
        return {"reply": f"OpenRouter Error: {str(e)}"}