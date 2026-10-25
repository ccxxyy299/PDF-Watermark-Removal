"""Shared utility functions for PDF processing platform."""
import os
import re
import uuid
import shutil
import time
import zipfile
import io
from flask import send_file, jsonify

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.join(_BASE_DIR, 'tasks')
# Ensure TASK_DIR exists and is accessible; fall back to /tmp if needed
os.makedirs(TASK_DIR, exist_ok=True)
try:
    os.listdir(TASK_DIR)
except (PermissionError, OSError):
    import tempfile
    TASK_DIR = os.path.join(tempfile.gettempdir(), 'pdf_toolbox_tasks')
    os.makedirs(TASK_DIR, exist_ok=True)


def create_task():
    """Create a new task directory and return task_id."""
    _cleanup_old_tasks()
    task_id = uuid.uuid4().hex
    task_folder = os.path.join(TASK_DIR, task_id)
    os.makedirs(task_folder, exist_ok=True)
    return task_id


def get_task_folder(task_id):
    """Get the task folder path, returns None if invalid."""
    if not task_id or not re.match(r'^[a-f0-9]{32}$', task_id):
        return None
    folder = os.path.join(TASK_DIR, task_id)
    if not os.path.isdir(folder):
        return None
    return folder


def parse_page_ranges(range_str, total_pages):
    """Parse page range string like '1-3,5,7-9' into sorted list of 0-indexed page numbers.

    Returns (pages, error_msg). pages is None on error.
    """
    if not range_str or range_str.strip().lower() in ('all', '全部', ''):
        return list(range(total_pages)), None

    pages = set()
    parts = re.split(r'[,，\s]+', range_str.strip())
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            match = re.match(r'^(\d+)-(\d+)$', part)
            if not match:
                return None, f'无效的页码范围: "{part}"'
            start = int(match.group(1))
            end = int(match.group(2))
            if start < 1 or end > total_pages or start > end:
                return None, f'页码范围 {start}-{end} 超出有效范围 (1-{total_pages})'
            for p in range(start - 1, end):
                pages.add(p)
        else:
            try:
                p = int(part)
            except ValueError:
                return None, f'无效的页码: "{part}"'
            if p < 1 or p > total_pages:
                return None, f'页码 {p} 超出有效范围 (1-{total_pages})'
            pages.add(p - 1)

    if not pages:
        return None, '未指定任何页面'
    return sorted(pages), None


def format_file_size(size_bytes):
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f'{size_bytes} B'
    elif size_bytes < 1024 * 1024:
        return f'{size_bytes / 1024:.1f} KB'
    else:
        return f'{size_bytes / (1024 * 1024):.2f} MB'


def send_zip_from_dir(directory, file_list, zip_name, filename_prefix=''):
    """Create a ZIP from files in directory and send it.

    file_list: list of filenames relative to directory
    filename_prefix: prefix for ZIP comment, not used for naming
    """
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in file_list:
            fpath = os.path.join(directory, fname)
            if os.path.exists(fpath):
                zf.write(fpath, fname)
    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=zip_name
    )


def _cleanup_old_tasks(max_age_hours=1):
    """Clean up task directories older than max_age_hours."""
    if not os.path.isdir(TASK_DIR):
        return
    now = time.time()
    for name in os.listdir(TASK_DIR):
        path = os.path.join(TASK_DIR, name)
        if os.path.isdir(path) and now - os.path.getmtime(path) > max_age_hours * 3600:
            shutil.rmtree(path, ignore_errors=True)


def get_pdf_info(pdf_path):
    """Get PDF page count and file size. Returns (page_count, file_size_bytes)."""
    import fitz
    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    doc.close()
    file_size = os.path.getsize(pdf_path)
    return page_count, file_size
