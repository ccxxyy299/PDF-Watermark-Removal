"""PDF to Image conversion routes."""
import os
import cv2
import numpy as np
import fitz
from flask import Blueprint, request, jsonify, send_file
from utils import (
    create_task, get_task_folder, parse_page_ranges,
    format_file_size, send_zip_from_dir
)

pdf2img_bp = Blueprint('pdf2img', __name__)


@pdf2img_bp.route('/pdf2img/upload', methods=['POST'])
def upload():
    uploaded_file = request.files.get('file')
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({'status': 'error', 'message': '请选择PDF文件'})
    if not uploaded_file.filename.lower().endswith('.pdf'):
        return jsonify({'status': 'error', 'message': '请上传PDF格式文件'})

    task_id = create_task()
    task_folder = get_task_folder(task_id)
    pdf_path = os.path.join(task_folder, 'input.pdf')
    uploaded_file.save(pdf_path)

    try:
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        doc.close()
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'PDF文件损坏或无法打开: {str(e)}'})

    # Save original filename (without extension) for output naming
    orig_name = os.path.splitext(uploaded_file.filename)[0]
    with open(os.path.join(task_folder, 'orig_name.txt'), 'w') as f:
        f.write(orig_name)

    return jsonify({
        'status': 'ok',
        'task_id': task_id,
        'page_count': page_count,
        'filename': uploaded_file.filename
    })


@pdf2img_bp.route('/pdf2img/convert', methods=['POST'])
def convert():
    data = request.get_json(silent=True) or request.form
    task_id = data.get('task_id', '')
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效的任务，请重新上传'})

    pdf_path = os.path.join(task_folder, 'input.pdf')
    if not os.path.exists(pdf_path):
        return jsonify({'status': 'error', 'message': '请先上传PDF文件'})

    fmt = data.get('format', 'png').lower()
    if fmt not in ('png', 'jpeg', 'jpg', 'webp'):
        fmt = 'png'
    ext = 'jpg' if fmt == 'jpeg' else fmt

    dpi = int(data.get('dpi', 300))
    if dpi not in (72, 150, 300, 600):
        dpi = 300

    quality = int(data.get('quality', 90))
    quality = max(1, min(100, quality))

    page_range_str = data.get('page_range', 'all')
    preview_page = data.get('preview_page', None)

    try:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count

        # Determine which pages to convert
        if page_range_str == 'current' and preview_page is not None:
            pages = [int(preview_page)]
        else:
            pages, err = parse_page_ranges(page_range_str, total_pages)
            if err:
                doc.close()
                return jsonify({'status': 'error', 'message': err})

        scale = dpi / 72.0
        output_dir = os.path.join(task_folder, f'images_{fmt}_{dpi}')
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(task_folder, 'orig_name.txt'), 'r') as f:
            orig_name = f.read().strip()

        converted_files = []
        for page_idx in pages:
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            img_array = np.frombuffer(pix.tobytes('png'), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            fname = f'{orig_name}_page_{page_idx + 1:03d}.{ext}'
            fpath = os.path.join(output_dir, fname)

            if ext == 'jpg':
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif ext == 'webp':
                encode_params = [cv2.IMWRITE_WEBP_QUALITY, quality]
            else:
                encode_params = []

            success, encoded = cv2.imencode('.' + ext, img, encode_params) if encode_params else cv2.imencode('.' + ext, img)
            if success:
                with open(fpath, 'wb') as f:
                    f.write(encoded.tobytes())
                converted_files.append(fname)

        doc.close()

        # Generate thumbnails for first 5 pages
        thumb_dir = os.path.join(task_folder, 'thumbs')
        os.makedirs(thumb_dir, exist_ok=True)
        preview_files = []
        doc2 = fitz.open(pdf_path)
        thumb_scale = 100 / 72.0  # ~100 DPI for thumbs
        for i, page_idx in enumerate(pages[:5]):
            page = doc2[page_idx]
            pix = page.get_pixmap(matrix=fitz.Matrix(thumb_scale, thumb_scale))
            tn = f'thumb_{page_idx + 1:03d}.jpg'
            tn_path = os.path.join(thumb_dir, tn)
            pix.save(tn_path, jpg_quality=70)
            preview_files.append(tn)
        doc2.close()

        # Save conversion metadata
        import json
        meta = {
            'format': fmt,
            'ext': ext,
            'dpi': dpi,
            'output_dir': output_dir,
            'files': converted_files,
            'orig_name': orig_name,
            'total_converted': len(converted_files)
        }
        with open(os.path.join(task_folder, 'convert_meta.json'), 'w') as f:
            json.dump(meta, f)

        return jsonify({
            'status': 'ok',
            'message': f'转换完成，共 {len(converted_files)} 页',
            'total_pages': len(converted_files),
            'preview_files': preview_files,
            'task_id': task_id
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'转换失败: {str(e)}'})


@pdf2img_bp.route('/pdf2img/thumb/<task_id>/<filename>')
def thumb(task_id, filename):
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效任务'})
    fpath = os.path.join(task_folder, 'thumbs', filename)
    if not os.path.exists(fpath):
        return jsonify({'status': 'error', 'message': '缩略图不存在'})
    return send_file(fpath, mimetype='image/jpeg')


@pdf2img_bp.route('/pdf2img/download_page')
def download_page():
    task_id = request.args.get('task_id', '')
    filename = request.args.get('file', '')
    task_folder = get_task_folder(task_id)
    if not task_folder or not filename:
        return jsonify({'status': 'error', 'message': '参数错误'})

    import json
    meta_path = os.path.join(task_folder, 'convert_meta.json')
    if not os.path.exists(meta_path):
        return jsonify({'status': 'error', 'message': '请先完成转换'})
    with open(meta_path) as f:
        meta = json.load(f)

    fpath = os.path.join(meta['output_dir'], filename)
    if not os.path.exists(fpath):
        return jsonify({'status': 'error', 'message': '文件不存在'})

    mime_map = {'png': 'image/png', 'jpg': 'image/jpeg', 'webp': 'image/webp'}
    ext = meta['ext']
    return send_file(
        fpath,
        mimetype=mime_map.get(ext, 'application/octet-stream'),
        as_attachment=True,
        download_name=filename
    )


@pdf2img_bp.route('/pdf2img/download_zip')
def download_zip():
    task_id = request.args.get('task_id', '')
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效任务'})

    import json
    meta_path = os.path.join(task_folder, 'convert_meta.json')
    if not os.path.exists(meta_path):
        return jsonify({'status': 'error', 'message': '请先完成转换'})
    with open(meta_path) as f:
        meta = json.load(f)

    zip_name = f"{meta['orig_name']}_images.zip"
    return send_zip_from_dir(meta['output_dir'], meta['files'], zip_name)


@pdf2img_bp.route('/pdf2img/file_list')
def file_list():
    task_id = request.args.get('task_id', '')
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效任务'})
    import json
    meta_path = os.path.join(task_folder, 'convert_meta.json')
    if not os.path.exists(meta_path):
        return jsonify({'status': 'error', 'message': '未转换'})
    with open(meta_path) as f:
        meta = json.load(f)
    return jsonify({'status': 'ok', 'files': meta['files']})
