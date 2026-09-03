import os
import io
import re
import json
import csv
import uuid
import time
import hmac
import hashlib
import mimetypes
import tempfile
from pathlib import Path
from datetime import datetime
from decimal import Decimal, InvalidOperation
from urllib.parse import urlparse

import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from google import genai


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

load_dotenv()

APP_TITLE = "GenAI Intelligent Invoice Vision Assistant"

# Use a stable Gemini model by default.
# Can be changed in .env.
MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)

# Prompt version is recorded in audit history.
PROMPT_VERSION = "invoice-assistant-v1.0"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

SUPPORTED_FILES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".pdf": "application/pdf",
}

AUDIT_FILE = Path("invoice_audit.json")


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧾",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "document_id": None,
    "file_name": None,
    "file_bytes": None,
    "mime_type": None,
    "file_hash": None,
    "upload_time": None,

    "invoice_data": None,
    "validation": None,

    "qa_history": [],

    "data_status": "NOT_PROCESSED",

    "extraction_error": None,

    "last_model_used": None,

    "last_ai_request": None,
    "last_ai_response": None,

    "audit_session_events": [],
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# GENERAL UTILITIES
# ============================================================

def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def safe_text(value):
    if value is None:
        return ""

    return str(value).strip()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def generate_document_id(file_hash):
    return "INV-" + file_hash[:16].upper()


def decimal_value(value):
    """
    Safely converts strings such as:

    ₹1,234.50
    $1,234.50
    1234.50

    into Decimal.
    """

    if value is None:
        return None

    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value))
        except Exception:
            return None

    text = str(value).strip()

    if not text:
        return None

    text = text.replace(",", "")

    # Keep digits, decimal point and minus sign.
    text = re.sub(r"[^\d.\-]", "", text)

    if not text:
        return None

    try:
        return Decimal(text)
    except InvalidOperation:
        return None


# ============================================================
# AUDIT HISTORY
# ============================================================

def load_audit_history():
    """
    Loads persistent audit history from invoice_audit.json.
    """

    try:

        if not AUDIT_FILE.exists():
            return []

        with open(
            AUDIT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


def save_audit_event(
    event_type,
    details=None,
    ai_request=None,
    ai_response=None,
    validation_result=None,
):
    """
    Requirement 5:
    Stores complete audit information including:

    - event
    - timestamp
    - document ID
    - file name
    - model
    - prompt version
    - AI request
    - AI response
    - validation result
    """

    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": now_iso(),

        "event_type": event_type,

        "document_id": st.session_state.get(
            "document_id"
        ),

        "file_name": st.session_state.get(
            "file_name"
        ),

        "file_hash": st.session_state.get(
            "file_hash"
        ),

        "model": st.session_state.get(
            "last_model_used"
        ) or MODEL_NAME,

        "prompt_version": PROMPT_VERSION,

        "ai_request": ai_request,

        "ai_response": ai_response,

        "validation_result": validation_result,

        "details": details or {},
    }

    history = load_audit_history()

    history.append(event)

    # Prevent unlimited growth.
    history = history[-3000:]

    try:

        with open(
            AUDIT_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                history,
                file,
                indent=2,
                ensure_ascii=False,
            )

    except Exception as error:

        st.warning(
            f"Audit history could not be saved: {error}"
        )

    st.session_state.audit_session_events.append(event)


# ============================================================
# FILE VALIDATION
# ============================================================

def detect_mime_type(file_name):

    extension = Path(
        file_name
    ).suffix.lower()

    if extension in SUPPORTED_FILES:
        return SUPPORTED_FILES[extension]

    guessed_type, _ = mimetypes.guess_type(
        file_name
    )

    return guessed_type


def validate_file_signature(
    file_bytes,
    mime_type,
):
    """
    Checks actual file signature.

    JPEG  -> FF D8 FF
    PNG   -> PNG signature
    PDF   -> %PDF
    """

    if mime_type == "image/jpeg":
        return file_bytes.startswith(
            b"\xff\xd8\xff"
        )

    if mime_type == "image/png":
        return file_bytes.startswith(
            b"\x89PNG\r\n\x1a\n"
        )

    if mime_type == "application/pdf":
        return file_bytes.startswith(
            b"%PDF"
        )

    return False


def validate_uploaded_file(
    uploaded_file
):
    """
    Requirement 1 + 2:
    Validates:

    - extension
    - file size
    - MIME type
    - file signature
    """

    if uploaded_file is None:

        return (
            False,
            "No file selected."
        )

    file_name = uploaded_file.name

    file_bytes = uploaded_file.getvalue()

    extension = Path(
        file_name
    ).suffix.lower()

    if extension not in SUPPORTED_FILES:

        return (
            False,
            "Unsupported format. "
            "Please upload JPG, JPEG, PNG or PDF."
        )

    if len(file_bytes) > MAX_FILE_SIZE:

        return (
            False,
            "File exceeds the maximum "
            "allowed size of 10 MB."
        )

    mime_type = detect_mime_type(
        file_name
    )

    if mime_type not in SUPPORTED_FILES.values():

        return (
            False,
            "Unsupported MIME type."
        )

    if not validate_file_signature(
        file_bytes,
        mime_type
    ):

        return (
            False,
            "File signature does not match "
            "the selected file type."
        )

    return (
        True,
        "File validation successful."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_api_key():

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        return None

    return api_key.strip()


def get_gemini_client():

    api_key = get_api_key()

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY is not configured. "
            "Add your Gemini API key to .env "
            "and restart Streamlit."
        )

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# TEMPORARY FILE
# ============================================================

def create_temp_file(
    file_bytes,
    file_name
):

    suffix = Path(
        file_name
    ).suffix.lower()

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temp_file.write(file_bytes)

    temp_file.flush()

    temp_file.close()

    return temp_file.name


# ============================================================
# GEMINI API CALL
# ============================================================

def call_gemini_with_file(
    file_bytes,
    file_name,
    prompt,
):
    """
    Sends the actual invoice file to Gemini.

    Includes:
    - temporary upload
    - retry for temporary errors
    - model fallback
    - AI request/response capture
    """

    client = get_gemini_client()

    temp_path = create_temp_file(
        file_bytes,
        file_name
    )

    # Remove duplicates while preserving order.
    models_to_try = list(
        dict.fromkeys(
            [
                MODEL_NAME,
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
            ]
        )
    )

    last_error = None

    try:

        for model in models_to_try:

            for attempt in range(2):

                try:

                    uploaded_file = client.files.upload(
                        file=temp_path
                    )

                    response = client.models.generate_content(
                        model=model,
                        contents=[
                            uploaded_file,
                            prompt,
                        ],
                    )

                    response_text = getattr(
                        response,
                        "text",
                        None
                    )

                    if not response_text:

                        raise RuntimeError(
                            "Gemini returned an empty response."
                        )

                    st.session_state.last_model_used = model

                    st.session_state.last_ai_request = prompt

                    st.session_state.last_ai_response = response_text

                    return response_text

                except Exception as error:

                    last_error = error

                    error_text = str(
                        error
                    ).lower()

                    temporary_error = any(
                        phrase in error_text
                        for phrase in [
                            "503",
                            "unavailable",
                            "high demand",
                            "overloaded",
                            "temporarily",
                            "429",
                            "resource exhausted",
                        ]
                    )

                    if temporary_error:

                        if attempt == 0:

                            time.sleep(2)

                            continue

                        # Try next model.
                        break

                    # Non-temporary error.
                    raise error

        raise RuntimeError(
            "Gemini service is temporarily unavailable. "
            f"Last error: {last_error}"
        )

    finally:

        try:
            os.remove(temp_path)
        except Exception:
            pass


# ============================================================
# JSON PARSING
# ============================================================

def extract_json_text(text):

    if not text:

        raise ValueError(
            "Gemini returned an empty response."
        )

    cleaned = text.strip()

    # Remove markdown fences.
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = cleaned.strip()

    # Direct JSON.
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Extract JSON object.
    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")

    if (
        object_start != -1
        and object_end > object_start
    ):

        candidate = cleaned[
            object_start:object_end + 1
        ]

        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Extract JSON array.
    array_start = cleaned.find("[")
    array_end = cleaned.rfind("]")

    if (
        array_start != -1
        and array_end > array_start
    ):

        candidate = cleaned[
            array_start:array_end + 1
        ]

        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError(
        "Gemini response could not be parsed as JSON."
    )


# ============================================================
# STRUCTURED DATA NORMALIZATION
# ============================================================

def normalize_invoice_data(data):

    # Gemini occasionally returns an array.
    if isinstance(data, list):

        if (
            len(data) == 1
            and isinstance(data[0], dict)
        ):

            data = data[0]

        else:

            data = {
                "supplier_name": None,
                "invoice_number": None,
                "invoice_date": None,
                "due_date": None,
                "currency": None,
                "subtotal": None,
                "tax": None,
                "total": None,
                "po_number": None,
                "line_items": data,
            }

    if not isinstance(data, dict):
        data = {}

    supplier_name = (
        data.get("supplier_name")
        or data.get("vendor_name")
        or data.get("supplier")
        or data.get("vendor")
    )

    invoice_number = (
        data.get("invoice_number")
        or data.get("invoice_no")
        or data.get("invoice_id")
    )

    invoice_date = (
        data.get("invoice_date")
        or data.get("date")
    )

    due_date = (
        data.get("due_date")
        or data.get("payment_due_date")
    )

    currency = data.get(
        "currency"
    )

    subtotal = (
        data.get("subtotal")
        or data.get("sub_total")
    )

    tax = (
        data.get("tax")
        or data.get("tax_amount")
        or data.get("total_tax")
    )

    total = (
        data.get("total")
        or data.get("grand_total")
        or data.get("total_amount")
        or data.get("amount_payable")
        or data.get("amount_due")
        or data.get("current_charges")
    )

    po_number = (
        data.get("po_number")
        or data.get("purchase_order_number")
        or data.get("po")
    )

    line_items = data.get(
        "line_items",
        []
    )

    if not isinstance(
        line_items,
        list
    ):

        line_items = []

    normalized_items = []

    for item in line_items:

        if not isinstance(
            item,
            dict
        ):
            continue

        description = (
            item.get("description")
            or item.get("item")
            or item.get("name")
            or ""
        )

        quantity = (
            item.get("quantity")
            if item.get("quantity") is not None
            else item.get("qty")
        )

        unit_price = (
            item.get("unit_price")
            if item.get("unit_price") is not None
            else item.get("price")
        )

        amount = (
            item.get("amount")
            if item.get("amount") is not None
            else item.get("line_total")
        )

        normalized_items.append(
            {
                "description": description,
                "quantity": quantity,
                "unit_price": unit_price,
                "amount": amount,
            }
        )

    return {
        "supplier_name": supplier_name,
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "due_date": due_date,
        "currency": currency,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "po_number": po_number,
        "line_items": normalized_items,
    }


# ============================================================
# REQUIREMENT 3
# INVOICE EXTRACTION PROMPT
# ============================================================

EXTRACTION_PROMPT = """
You are an expert invoice/document extraction AI.

Analyze the uploaded invoice, supplier invoice, tax invoice,
utility bill, electricity bill, service invoice, or similar
financial document.

Return EXACTLY ONE JSON OBJECT.
Do not return markdown.
Do not return explanations outside JSON.

Use exactly this structure:

{
  "supplier_name": null,
  "invoice_number": null,
  "invoice_date": null,
  "due_date": null,
  "currency": null,
  "subtotal": null,
  "tax": null,
  "total": null,
  "po_number": null,
  "line_items": [
    {
      "description": null,
      "quantity": null,
      "unit_price": null,
      "amount": null
    }
  ]
}

RULES:

1. Extract ONLY information visible in the document.
2. Never invent or guess information.
3. Use null when a value is unavailable or unreadable.
4. Preserve invoice numbers exactly as text.
5. Preserve dates as visible where practical.
6. Extract currency when visible.
7. Extract subtotal when visible.
8. Extract tax when visible.
9. Extract final payable total when visible.
10. Recognize labels such as:
    - Grand Total
    - Total
    - Total Amount
    - Amount Payable
    - Amount Due
    - Net Payable
    - Current Charges
    - Total Due
11. Extract purchase order number when visible.
12. Extract all clearly visible line items.
13. For utility bills, quantity and unit price can be null.
14. Utility/service charges can still be line items with
    description and amount.
15. Do not confuse invoice date and due date.
16. Do not calculate missing fields unless the document
    explicitly provides enough information.
17. Do not use external knowledge.
"""


# ============================================================
# EXTRACT INVOICE
# ============================================================

def extract_invoice_data(
    file_bytes,
    file_name,
):

    raw_response = call_gemini_with_file(
        file_bytes,
        file_name,
        EXTRACTION_PROMPT,
    )

    parsed = extract_json_text(
        raw_response
    )

    normalized = normalize_invoice_data(
        parsed
    )

    return (
        normalized,
        raw_response,
    )


# ============================================================
# REQUIREMENT 3
# NATURAL LANGUAGE Q&A
# ============================================================

def ask_invoice_question(
    file_bytes,
    file_name,
    question,
    history,
):

    previous_context = ""

    if history:

        previous_context = (
            "\nPrevious conversation context:\n"
        )

        for item in history[-5:]:

            previous_context += (
                f"User: {item.get('question', '')}\n"
                f"Assistant: {item.get('answer', '')}\n"
            )

    prompt = f"""
You are a professional invoice knowledge assistant.

Answer the user's question using ONLY information visible
in the uploaded invoice/document.

USER QUESTION:
{question}

{previous_context}

RULES:

1. Do not invent information.
2. Do not use external information.
3. If information is not available, say:
   "This information is not available in the uploaded document."
4. For numerical questions, use values from the document.
5. If arithmetic is required, show the calculation briefly.
6. For follow-up questions, use previous conversation context
   only when relevant.
7. Keep answers clear and concise.
"""

    answer = call_gemini_with_file(
        file_bytes,
        file_name,
        prompt,
    )

    return (
        answer,
        prompt,
    )


# ============================================================
# REQUIREMENT 4
# VALIDATION
# ============================================================

def validate_invoice_data(data):

    checks = []

    subtotal = decimal_value(
        data.get("subtotal")
    )

    tax = decimal_value(
        data.get("tax")
    )

    total = decimal_value(
        data.get("total")
    )

    # --------------------------------------------------------
    # TOTAL CHECK
    # --------------------------------------------------------

    if total is None:

        checks.append(
            {
                "check": "Total Check",
                "status": "WARNING",
                "message": (
                    "Final invoice total could not "
                    "be identified clearly."
                ),
            }
        )

    elif (
        subtotal is not None
        and tax is not None
    ):

        expected_total = (
            subtotal + tax
        )

        difference = abs(
            expected_total - total
        )

        if difference <= Decimal("0.02"):

            checks.append(
                {
                    "check": "Total Check",
                    "status": "PASS",
                    "message": (
                        f"Subtotal + Tax = "
                        f"{expected_total}; "
                        f"Total = {total}. "
                        "Values are consistent."
                    ),
                }
            )

        else:

            checks.append(
                {
                    "check": "Total Check",
                    "status": "WARNING",
                    "message": (
                        f"Subtotal + Tax = "
                        f"{expected_total}, "
                        f"but Total = {total}. "
                        f"Difference = {difference}."
                    ),
                }
            )

    else:

        checks.append(
            {
                "check": "Total Check",
                "status": "INFO",
                "message": (
                    "Final total was identified, "
                    "but subtotal and/or tax were "
                    "not available for arithmetic verification."
                ),
            }
        )

    # --------------------------------------------------------
    # REQUIRED FIELDS
    # --------------------------------------------------------

    required_fields = {
        "Supplier Name": data.get(
            "supplier_name"
        ),
        "Invoice Number": data.get(
            "invoice_number"
        ),
        "Invoice Date": data.get(
            "invoice_date"
        ),
    }

    missing = [
        field
        for field, value in required_fields.items()
        if not value
    ]

    if not missing:

        checks.append(
            {
                "check": "Required Fields",
                "status": "PASS",
                "message": (
                    "Supplier name, invoice number "
                    "and invoice date are available."
                ),
            }
        )

    else:

        checks.append(
            {
                "check": "Required Fields",
                "status": "WARNING",
                "message": (
                    "Missing or unclear fields: "
                    + ", ".join(missing)
                ),
            }
        )

    # --------------------------------------------------------
    # LINE ITEMS
    # --------------------------------------------------------

    items = data.get(
        "line_items",
        []
    )

    if not items:

        checks.append(
            {
                "check": "Line Item Check",
                "status": "INFO",
                "message": (
                    "No line items were clearly identified."
                ),
            }
        )

    else:

        amount_count = 0
        arithmetic_count = 0
        arithmetic_errors = []

        for index, item in enumerate(
            items,
            start=1
        ):

            amount = decimal_value(
                item.get("amount")
            )

            quantity = decimal_value(
                item.get("quantity")
            )

            unit_price = decimal_value(
                item.get("unit_price")
            )

            if amount is not None:

                amount_count += 1

            if (
                quantity is not None
                and unit_price is not None
                and amount is not None
            ):

                arithmetic_count += 1

                expected = (
                    quantity * unit_price
                )

                if abs(
                    expected - amount
                ) > Decimal("0.02"):

                    arithmetic_errors.append(
                        f"Line {index}: "
                        f"expected {expected}, "
                        f"found {amount}"
                    )

        if arithmetic_errors:

            checks.append(
                {
                    "check": "Line Item Check",
                    "status": "WARNING",
                    "message": "; ".join(
                        arithmetic_errors
                    ),
                }
            )

        elif amount_count > 0:

            checks.append(
                {
                    "check": "Line Item Check",
                    "status": "PASS",
                    "message": (
                        f"{amount_count} line item(s) "
                        "have amounts. "
                        f"{arithmetic_count} line item(s) "
                        "also had sufficient numeric data "
                        "for quantity × unit price validation."
                    ),
                }
            )

        else:

            checks.append(
                {
                    "check": "Line Item Check",
                    "status": "INFO",
                    "message": (
                        "Line items were detected but "
                        "numeric amounts were unavailable."
                    ),
                }
            )

    # --------------------------------------------------------
    # OVERALL STATUS
    # --------------------------------------------------------

    if any(
        item["status"] == "WARNING"
        for item in checks
    ):

        overall = "WARNING"

    else:

        overall = "PASS"

    return {
        "overall_status": overall,
        "checks": checks,
        "validated_at": now_iso(),
    }


# ============================================================
# UPDATE DATA STATUS
# ============================================================

def save_invoice_corrections(
    supplier_name,
    invoice_number,
    invoice_date,
    due_date,
    currency,
    subtotal,
    tax,
    total,
    po_number,
):

    data = st.session_state.invoice_data

    data.update(
        {
            "supplier_name": supplier_name or None,
            "invoice_number": invoice_number or None,
            "invoice_date": invoice_date or None,
            "due_date": due_date or None,
            "currency": currency or None,
            "subtotal": subtotal or None,
            "tax": tax or None,
            "total": total or None,
            "po_number": po_number or None,
        }
    )

    st.session_state.invoice_data = data

    st.session_state.validation = (
        validate_invoice_data(data)
    )


# ============================================================
# JSON EXPORT
# ============================================================

def create_json_export():

    return json.dumps(
        {
            "document_id": st.session_state.document_id,
            "file_name": st.session_state.file_name,
            "file_hash": st.session_state.file_hash,
            "uploaded_at": st.session_state.upload_time,
            "processed_at": now_iso(),

            "model": (
                st.session_state.last_model_used
                or MODEL_NAME
            ),

            "prompt_version": PROMPT_VERSION,

            "data_status": (
                st.session_state.data_status
            ),

            "invoice": (
                st.session_state.invoice_data
            ),

            "validation": (
                st.session_state.validation
            ),

            "qa_history": (
                st.session_state.qa_history
            ),
        },
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# CSV EXPORT
# ============================================================

def create_csv_export(data):

    output = io.StringIO()

    writer = csv.writer(output)

    writer.writerow(
        ["Field", "Value"]
    )

    fields = [
        (
            "Supplier Name",
            data.get("supplier_name")
        ),
        (
            "Invoice Number",
            data.get("invoice_number")
        ),
        (
            "Invoice Date",
            data.get("invoice_date")
        ),
        (
            "Due Date",
            data.get("due_date")
        ),
        (
            "Currency",
            data.get("currency")
        ),
        (
            "Subtotal",
            data.get("subtotal")
        ),
        (
            "Tax",
            data.get("tax")
        ),
        (
            "Total",
            data.get("total")
        ),
        (
            "PO Number",
            data.get("po_number")
        ),
    ]

    for field, value in fields:

        writer.writerow(
            [field, value]
        )

    writer.writerow([])

    writer.writerow(
        [
            "Description",
            "Quantity",
            "Unit Price",
            "Amount",
        ]
    )

    for item in data.get(
        "line_items",
        []
    ):

        writer.writerow(
            [
                item.get("description"),
                item.get("quantity"),
                item.get("unit_price"),
                item.get("amount"),
            ]
        )

    return output.getvalue()


# ============================================================
# INTEGRATION PAYLOAD
# ============================================================

def create_integration_payload():

    return {
        "event": "invoice.processed",

        "timestamp": now_iso(),

        "document": {
            "document_id": (
                st.session_state.document_id
            ),
            "file_name": (
                st.session_state.file_name
            ),
            "file_hash": (
                st.session_state.file_hash
            ),
        },

        "processing": {
            "model": (
                st.session_state.last_model_used
                or MODEL_NAME
            ),
            "prompt_version": PROMPT_VERSION,
        },

        "invoice": (
            st.session_state.invoice_data
        ),

        "validation": (
            st.session_state.validation
        ),

        "data_status": (
            st.session_state.data_status
        ),
    }


# ============================================================
# HMAC SIGNATURE
# ============================================================

def create_hmac_signature(
    payload,
    secret
):

    body = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()


# ============================================================
# WEBHOOK URL VALIDATION
# ============================================================

def validate_webhook_url(url):

    parsed = urlparse(
        url.strip()
    )

    if parsed.scheme not in (
        "http",
        "https",
    ):

        return False

    if not parsed.netloc:

        return False

    return True


# ============================================================
# OPTIONAL WEBHOOK SEND
# ============================================================

def send_webhook(
    webhook_url,
    payload,
    secret,
):
    """
    Sends an authenticated HMAC-SHA256 signed webhook.

    This is optional integration functionality.
    """

    import requests

    if not validate_webhook_url(
        webhook_url
    ):

        raise ValueError(
            "Invalid webhook URL."
        )

    signature = create_hmac_signature(
        payload,
        secret,
    )

    headers = {
        "Content-Type": "application/json",
        "X-Invoice-Signature": signature,
    }

    response = requests.post(
        webhook_url,
        json=payload,
        headers=headers,
        timeout=10,
    )

    response.raise_for_status()

    return response


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧾 Invoice Assistant")

    st.caption(
        "GenAI Intelligent Invoice Vision Assistant"
    )

    st.divider()

    st.markdown(
        "### 5 Project Requirements"
    )

    st.markdown(
        """
**1.** Upload & Preview

**2.** Validation & Processing

**3.** Gemini Q&A

**4.** Structured Extraction & Validation

**5.** Audit, History, Export & Integration
"""
    )

    st.divider()

    st.caption(
        f"Gemini: {MODEL_NAME}"
    )

    st.caption(
        f"Prompt: {PROMPT_VERSION}"
    )

    st.caption(
        "No login/signup required"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "🧾 GenAI Intelligent Invoice Vision Assistant"
)

st.write(
    "Upload an invoice, extract structured information, "
    "validate the extracted data, ask questions, "
    "and export or integrate the result."
)


# ============================================================
# REQUIREMENT 1
# UPLOAD & PREVIEW
# ============================================================

st.header(
    "1️⃣ Invoice Upload & Preview"
)

uploaded_file = st.file_uploader(
    "Upload Invoice",
    type=[
        "jpg",
        "jpeg",
        "png",
        "pdf",
    ],
    help=(
        "Supported: JPG, JPEG, PNG and PDF. "
        "Maximum size: 10 MB."
    ),
)


if uploaded_file:

    valid, message = (
        validate_uploaded_file(
            uploaded_file
        )
    )

    if not valid:

        st.error(message)

        save_audit_event(
            "FILE_VALIDATION_FAILED",
            {
                "reason": message,
            },
        )

    else:

        file_bytes = (
            uploaded_file.getvalue()
        )

        file_hash = sha256_bytes(
            file_bytes
        )

        document_id = generate_document_id(
            file_hash
        )

        mime_type = detect_mime_type(
            uploaded_file.name
        )

        # Detect new upload.
        new_document = (
            st.session_state.file_hash
            != file_hash
        )

        if new_document:

            st.session_state.document_id = (
                document_id
            )

            st.session_state.file_name = (
                uploaded_file.name
            )

            st.session_state.file_bytes = (
                file_bytes
            )

            st.session_state.mime_type = (
                mime_type
            )

            st.session_state.file_hash = (
                file_hash
            )

            st.session_state.upload_time = (
                now_iso()
            )

            st.session_state.invoice_data = None

            st.session_state.validation = None

            st.session_state.qa_history = []

            st.session_state.data_status = (
                "UPLOADED"
            )

            st.session_state.extraction_error = (
                None
            )

            st.session_state.last_model_used = (
                None
            )

            st.session_state.last_ai_request = (
                None
            )

            st.session_state.last_ai_response = (
                None
            )

            save_audit_event(
                "DOCUMENT_UPLOADED",
                {
                    "file_name": uploaded_file.name,
                    "mime_type": mime_type,
                    "size_bytes": len(file_bytes),
                    "sha256": file_hash,
                },
            )

        st.success(
            "File validated successfully."
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Document ID",
                st.session_state.document_id,
            )

        with c2:

            st.metric(
                "File Size",
                f"{len(file_bytes) / 1024:.1f} KB",
            )

        with c3:

            st.metric(
                "Status",
                st.session_state.data_status,
            )

        # ----------------------------------------------------
        # IMAGE PREVIEW
        # ----------------------------------------------------

        if mime_type in (
            "image/jpeg",
            "image/png",
        ):

            try:

                image = Image.open(
                    io.BytesIO(file_bytes)
                )

                st.image(
                    image,
                    caption=uploaded_file.name,
                    width="stretch",
                )

            except Exception as error:

                st.error(
                    f"Image preview failed: {error}"
                )

        # ----------------------------------------------------
        # PDF PREVIEW
        # ----------------------------------------------------

        elif mime_type == "application/pdf":

            st.subheader(
                "PDF Preview"
            )

            # Browser-based PDF viewer.
            pdf_base64 = __import__(
                "base64"
            ).b64encode(
                file_bytes
            ).decode("utf-8")

            pdf_html = f"""
            <iframe
                src="data:application/pdf;base64,{pdf_base64}"
                width="100%"
                height="650"
                style="border:1px solid #ccc;"
            ></iframe>
            """

            st.components.v1.html(
                pdf_html,
                height=670,
                scrolling=True,
            )


# ============================================================
# REQUIREMENT 2
# DOCUMENT PROCESSING
# ============================================================

if st.session_state.file_bytes:

    st.header(
        "2️⃣ Invoice Validation & Document Processing"
    )

    st.write(
        "The document has passed file validation. "
        "Click below to send it to Gemini for multimodal "
        "invoice processing and structured extraction."
    )

    if st.button(
        "🔍 Extract Invoice Data",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.extraction_error = None

        with st.spinner(
            "Processing invoice with Gemini..."
        ):

            try:

                data, raw_response = (
                    extract_invoice_data(
                        st.session_state.file_bytes,
                        st.session_state.file_name,
                    )
                )

                validation = (
                    validate_invoice_data(
                        data
                    )
                )

                st.session_state.invoice_data = data

                st.session_state.validation = (
                    validation
                )

                st.session_state.data_status = (
                    "AI_GENERATED"
                )

                save_audit_event(
                    "AI_EXTRACTION_COMPLETED",
                    {
                        "validation_status": (
                            validation[
                                "overall_status"
                            ]
                        ),
                    },
                    ai_request=(
                        st.session_state.last_ai_request
                    ),
                    ai_response=raw_response,
                    validation_result=validation,
                )

                st.success(
                    "Invoice extraction completed successfully."
                )

            except Exception as error:

                st.session_state.extraction_error = (
                    str(error)
                )

                st.session_state.data_status = (
                    "EXTRACTION_FAILED"
                )

                save_audit_event(
                    "AI_EXTRACTION_FAILED",
                    {
                        "error": str(error),
                    },
                    ai_request=(
                        st.session_state.last_ai_request
                    ),
                    ai_response=(
                        st.session_state.last_ai_response
                    ),
                )

                st.error(
                    "Invoice extraction failed."
                )

                st.code(
                    str(error),
                    language="text",
                )


# ============================================================
# REQUIREMENT 3
# NATURAL LANGUAGE Q&A
# ============================================================

if st.session_state.file_bytes:

    st.header(
        "3️⃣ Knowledge Assistant — Invoice Q&A"
    )

    question = st.text_input(
        "Ask a question about the invoice",
        placeholder=(
            "Example: What is the total amount payable?"
        ),
    )

    if st.button(
        "💬 Ask Gemini",
        use_container_width=True,
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Analyzing your question..."
            ):

                try:

                    answer, qa_prompt = (
                        ask_invoice_question(
                            st.session_state.file_bytes,
                            st.session_state.file_name,
                            question.strip(),
                            st.session_state.qa_history,
                        )
                    )

                    record = {
                        "timestamp": now_iso(),
                        "question": question.strip(),
                        "answer": answer,
                        "model": (
                            st.session_state.last_model_used
                            or MODEL_NAME
                        ),
                        "prompt_version": (
                            PROMPT_VERSION
                        ),
                    }

                    st.session_state.qa_history.append(
                        record
                    )

                    save_audit_event(
                        "AI_QUESTION_ANSWERED",
                        {
                            "question": question.strip(),
                        },
                        ai_request=qa_prompt,
                        ai_response=answer,
                    )

                    st.success(
                        "Answer"
                    )

                    st.write(answer)

                except Exception as error:

                    save_audit_event(
                        "AI_QUESTION_FAILED",
                        {
                            "question": question.strip(),
                            "error": str(error),
                        },
                        ai_request=(
                            st.session_state.last_ai_request
                        ),
                        ai_response=(
                            st.session_state.last_ai_response
                        ),
                    )

                    st.error(
                        f"Unable to answer: {error}"
                    )

    if st.session_state.qa_history:

        with st.expander(
            "💬 Conversation History"
        ):

            for item in reversed(
                st.session_state.qa_history
            ):

                st.markdown(
                    f"**You:** {item['question']}"
                )

                st.markdown(
                    f"**Assistant:** {item['answer']}"
                )

                st.divider()


# ============================================================
# REQUIREMENT 4
# STRUCTURED EXTRACTION
# ============================================================

if st.session_state.invoice_data:

    st.header(
        "4️⃣ Structured Invoice Data & Validation"
    )

    data = st.session_state.invoice_data

    col1, col2 = st.columns(2)

    with col1:

        supplier_name = st.text_input(
            "Supplier Name",
            value=safe_text(
                data.get("supplier_name")
            ),
            key="supplier_field",
        )

        invoice_number = st.text_input(
            "Invoice Number",
            value=safe_text(
                data.get("invoice_number")
            ),
            key="invoice_number_field",
        )

        invoice_date = st.text_input(
            "Invoice Date",
            value=safe_text(
                data.get("invoice_date")
            ),
            key="invoice_date_field",
        )

        due_date = st.text_input(
            "Due Date",
            value=safe_text(
                data.get("due_date")
            ),
            key="due_date_field",
        )

        po_number = st.text_input(
            "PO Number",
            value=safe_text(
                data.get("po_number")
            ),
            key="po_number_field",
        )

    with col2:

        currency = st.text_input(
            "Currency",
            value=safe_text(
                data.get("currency")
            ),
            key="currency_field",
        )

        subtotal = st.text_input(
            "Subtotal",
            value=safe_text(
                data.get("subtotal")
            ),
            key="subtotal_field",
        )

        tax = st.text_input(
            "Tax",
            value=safe_text(
                data.get("tax")
            ),
            key="tax_field",
        )

        total = st.text_input(
            "Total",
            value=safe_text(
                data.get("total")
            ),
            key="total_field",
        )

    # --------------------------------------------------------
    # LINE ITEMS
    # --------------------------------------------------------

    st.subheader(
        "Line Items"
    )

    items = data.get(
        "line_items",
        []
    )

    if items:

        for index, item in enumerate(
            items
        ):

            with st.container(
                border=True
            ):

                c1, c2, c3, c4 = st.columns(4)

                with c1:

                    st.write(
                        "**Description**"
                    )

                    st.write(
                        safe_text(
                            item.get(
                                "description"
                            )
                        )
                    )

                with c2:

                    st.write(
                        "**Quantity**"
                    )

                    st.write(
                        safe_text(
                            item.get(
                                "quantity"
                            )
                        )
                    )

                with c3:

                    st.write(
                        "**Unit Price**"
                    )

                    st.write(
                        safe_text(
                            item.get(
                                "unit_price"
                            )
                        )
                    )

                with c4:

                    st.write(
                        "**Amount**"
                    )

                    st.write(
                        safe_text(
                            item.get(
                                "amount"
                            )
                        )
                    )

    else:

        st.info(
            "No line items were clearly extracted."
        )

    # --------------------------------------------------------
    # DATA STATUS
    # --------------------------------------------------------

    st.info(
        f"Data Status: "
        f"**{st.session_state.data_status}**"
    )

    # --------------------------------------------------------
    # ACTION BUTTONS
    # --------------------------------------------------------

    b1, b2, b3 = st.columns(3)

    with b1:

        if st.button(
            "💾 Save Corrections",
            use_container_width=True,
        ):

            save_invoice_corrections(
                supplier_name,
                invoice_number,
                invoice_date,
                due_date,
                currency,
                subtotal,
                tax,
                total,
                po_number,
            )

            st.session_state.data_status = (
                "CORRECTED"
            )

            save_audit_event(
                "INVOICE_DATA_CORRECTED",
                {
                    "validation": (
                        st.session_state.validation
                    )
                },
                validation_result=(
                    st.session_state.validation
                ),
            )

            st.success(
                "Corrections saved successfully."
            )

    with b2:

        if st.button(
            "✅ Confirm Data",
            use_container_width=True,
        ):

            save_invoice_corrections(
                supplier_name,
                invoice_number,
                invoice_date,
                due_date,
                currency,
                subtotal,
                tax,
                total,
                po_number,
            )

            st.session_state.data_status = (
                "CONFIRMED"
            )

            save_audit_event(
                "INVOICE_DATA_CONFIRMED",
                {
                    "validation": (
                        st.session_state.validation
                    )
                },
                validation_result=(
                    st.session_state.validation
                ),
            )

            st.success(
                "Invoice data confirmed."
            )

    with b3:

        if st.button(
            "❌ Reject Data",
            use_container_width=True,
        ):

            st.session_state.data_status = (
                "REJECTED"
            )

            save_audit_event(
                "INVOICE_DATA_REJECTED",
                validation_result=(
                    st.session_state.validation
                ),
            )

            st.warning(
                "Invoice data marked as REJECTED."
            )


# ============================================================
# VALIDATION DISPLAY
# ============================================================

if st.session_state.validation:

    st.subheader(
        "Validation Results"
    )

    validation = (
        st.session_state.validation
    )

    overall = validation[
        "overall_status"
    ]

    if overall == "PASS":

        st.success(
            "Overall Validation: PASS"
        )

    else:

        st.warning(
            "Overall Validation: "
            "WARNING — review flagged items."
        )

    for result in validation[
        "checks"
    ]:

        status = result[
            "status"
        ]

        if status == "PASS":

            st.success(
                f"✅ {result['check']}: "
                f"{result['message']}"
            )

        elif status == "WARNING":

            st.warning(
                f"⚠️ {result['check']}: "
                f"{result['message']}"
            )

        else:

            st.info(
                f"ℹ️ {result['check']}: "
                f"{result['message']}"
            )


# ============================================================
# REQUIREMENT 5
# AUDIT / HISTORY / EXPORT / INTEGRATION
# ============================================================

if st.session_state.invoice_data:

    st.header(
        "5️⃣ Audit, History, Export & Integration"
    )

    data = st.session_state.invoice_data

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    st.subheader(
        "📤 Export"
    )

    json_data = create_json_export()

    csv_data = create_csv_export(
        data
    )

    e1, e2 = st.columns(2)

    with e1:

        st.download_button(
            "⬇️ Export JSON",
            data=json_data,
            file_name=(
                f"{st.session_state.document_id}.json"
            ),
            mime="application/json",
            use_container_width=True,
        )

    with e2:

        st.download_button(
            "⬇️ Export CSV",
            data=csv_data,
            file_name=(
                f"{st.session_state.document_id}.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    st.subheader(
        "🔎 Searchable Audit History"
    )

    audit_history = load_audit_history()

    search_term = st.text_input(
        "Search audit events",
        placeholder=(
            "Search document ID, event, filename, question..."
        ),
    )

    if search_term.strip():

        term = search_term.lower()

        audit_history = [
            event
            for event in audit_history
            if term in json.dumps(
                event,
                ensure_ascii=False,
            ).lower()
        ]

    if audit_history:

        for event in reversed(
            audit_history[-100:]
        ):

            st.markdown(
                f"""
**{event.get('event_type', 'UNKNOWN')}**

🕒 `{event.get('timestamp', '')}`

📄 Document: `{event.get('document_id', '')}`

🤖 Model: `{event.get('model', '')}`

📝 Prompt Version: `{event.get('prompt_version', '')}`
"""
            )

            with st.expander(
                "View complete audit record"
            ):

                st.json(
                    event
                )

    else:

        st.info(
            "No matching audit records found."
        )

    # --------------------------------------------------------
    # INTEGRATION
    # --------------------------------------------------------

    st.subheader(
        "🔗 ERP / Accounts Payable Integration"
    )

    st.write(
        "The application generates a structured "
        "webhook-ready payload for future ERP/AP integration."
    )

    webhook_url = st.text_input(
        "Webhook URL",
        placeholder=(
            "https://your-erp.example.com/webhook"
        ),
    )

    webhook_secret = st.text_input(
        "Webhook Secret",
        type="password",
        placeholder=(
            "Enter secret for HMAC-SHA256 authentication"
        ),
    )

    payload = create_integration_payload()

    if st.button(
        "Generate Integration Payload",
        use_container_width=True,
    ):

        if webhook_secret.strip():

            signature = create_hmac_signature(
                payload,
                webhook_secret.strip(),
            )

        else:

            signature = (
                "Secret not provided"
            )

        st.json(
            {
                "webhook_url": webhook_url,
                "authentication": "HMAC-SHA256",
                "headers": {
                    "Content-Type": "application/json",
                    "X-Invoice-Signature": signature,
                },
                "payload": payload,
            }
        )

        save_audit_event(
            "INTEGRATION_PAYLOAD_GENERATED",
            {
                "webhook_url": webhook_url,
                "authentication": "HMAC-SHA256",
            },
        )

    # --------------------------------------------------------
    # OPTIONAL SEND WEBHOOK
    # --------------------------------------------------------

    if (
        webhook_url.strip()
        and webhook_secret.strip()
    ):

        if st.button(
            "🚀 Send to Webhook",
            use_container_width=True,
        ):

            try:

                response = send_webhook(
                    webhook_url.strip(),
                    payload,
                    webhook_secret.strip(),
                )

                save_audit_event(
                    "WEBHOOK_SENT",
                    {
                        "url": webhook_url,
                        "http_status": response.status_code,
                    },
                )

                st.success(
                    f"Webhook sent successfully. "
                    f"HTTP {response.status_code}"
                )

            except Exception as error:

                save_audit_event(
                    "WEBHOOK_FAILED",
                    {
                        "url": webhook_url,
                        "error": str(error),
                    },
                )

                st.error(
                    f"Webhook failed: {error}"
                )


# ============================================================
# PROCESSING WORKFLOW
# ============================================================

st.divider()

st.header(
    "🔄 Complete Processing Workflow"
)

st.markdown(
    """
**Upload**
→ **File Validation**
→ **SHA-256 Document ID**
→ **Gemini Multimodal Processing**
→ **Structured Extraction**
→ **Data Validation**
→ **Correction / Confirmation / Rejection**
→ **Natural-Language Q&A**
→ **Audit History**
→ **JSON / CSV Export**
→ **ERP/AP Webhook Integration**
"""
)


# ============================================================
# REQUIREMENT COVERAGE
# ============================================================

with st.expander(
    "✅ Requirement Coverage"
):

    st.markdown(
        """
### Requirement 1 — Upload & Preview
- Streamlit upload interface
- JPG/JPEG/PNG/PDF
- Image preview
- PDF preview
- File status
- Error handling
- Question input

### Requirement 2 — Validation & Processing
- Extension validation
- MIME validation
- File-size validation
- File-signature validation
- SHA-256 hash
- Unique document ID
- Gemini document processing

### Requirement 3 — Gemini Q&A
- Gemini multimodal model
- Natural-language invoice questions
- Invoice-grounded answers
- No intentional hallucination/guessing
- Follow-up conversation context
- AI request/response auditing

### Requirement 4 — Structured Extraction
- Supplier name
- Invoice number
- Invoice date
- Due date
- Currency
- Subtotal
- Tax
- Total
- PO number
- Line items
- Quantity
- Unit price
- Amount
- Required-field validation
- Arithmetic validation
- AI_GENERATED status
- CORRECTED status
- CONFIRMED status
- REJECTED status

### Requirement 5 — Audit / History / Export / Integration
- Persistent audit history
- Document ID
- File hash
- Timestamp
- Event type
- AI request
- AI response
- Model version
- Prompt version
- Validation result
- Searchable history
- JSON export
- CSV export
- ERP/AP integration payload
- HMAC-SHA256 webhook authentication
- Optional webhook sending
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "GenAI Intelligent Invoice Vision Assistant | "
    "5-Requirement Implementation"
)
