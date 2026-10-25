"""PDF Compress routes."""
import os
import shutil
import cv2
import numpy as np
import fitz
from flask import Blueprint, request, jsonify, send_file
from utils import create_task, get_task_folder, format_file_size

pdfcompress_bp = Blueprint('pdfcompress', __name__)


@pdfcompress_bp.route('/pdfcompress/upload', methods=['POST'])
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
        return jsonify({'status': 'error', 'message': f'PDF文件损坏: {str(e)}'})

    file_size = os.path.getsize(pdf_path)
    orig_name = os.path.splitext(uploaded_file.filename)[0]
    with open(os.path.join(task_folder, 'orig_name.txt'), 'w') as f:
        f.write(orig_name)

    return jsonify({
        'status': 'ok',
        'task_id': task_id,
        'page_count': page_count,
        'file_size': file_size,
        'file_size_str': format_file_size(file_size),
        'filename': uploaded_file.filename
    })


@pdfcompress_bp.route('/pdfcompress/compress', methods=['POST'])
def compress():
    data = request.get_json(silent=True) or request.form
    task_id = data.get('task_id', '')
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效任务'})

    pdf_path = os.path.join(task_folder, 'input.pdf')
    if not os.path.exists(pdf_path):
        return jsonify({'status': 'error', 'message': '请先上传PDF文件'})

    level = data.get('level', 'standard')
    if level not in ('light', 'standard', 'strong'):
        level = 'standard'

    output_path = os.path.join(task_folder, 'output.pdf')
    original_size = os.path.getsize(pdf_path)

    try:
        if level == 'light':
            doc = fitz.open(pdf_path)
            doc.save(output_path, garbage=4, deflate=True)
            doc.close()

        elif level == 'standard':
            doc = fitz.open(pdf_path)
            try:
                doc.save(output_path, garbage=4, deflate=True, clean=True, linear=True)
            except Exception:
                # linear=True not supported in some PyMuPDF versions
                doc.save(output_path, garbage=4, deflate=True, clean=True)
            doc.close()

        elif level == 'strong':
            # Strong compression: render pages at 150 DPI, save as JPEG, rebuild PDF
            # This rasterizes all content and re-encodes as JPEG, dramatically reducing
            # file size for image-heavy PDFs while maintaining readability.
            src_doc = fitz.open(pdf_path)
            new_doc = fitz.open()
            target_dpi = 150
            scale = target_dpi / 72.0
            jpeg_quality = 75

            for page_num in range(src_doc.page_count):
                page = src_doc[page_num]
                pw = page.rect.width
                ph = page.rect.height
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                img_array = np.frombuffer(pix.tobytes('png'), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                # Encode as JPEG
                success, encoded = cv2.imencode(
                    '.jpg', img,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality, cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
                )
                if success:
                    new_page = new_doc.new_page(width=pw, height=ph)
                    new_page.insert_image(
                        fitz.Rect(0, 0, pw, ph),
                        stream=encoded.tobytes()
                    )
            try:
                new_doc.save(output_path, garbage=4, deflate=True, clean=True)
            except Exception:
                new_doc.save(output_path, garbage=4, deflate=True)
            new_doc.close()
            src_doc.close()

        compressed_size = os.path.getsize(output_path)
        ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

        if compressed_size >= original_size:
            shutil.copy2(pdf_path, output_path)
            return jsonify({
                'status': 'ok',
                'message': '该文件已无压缩空间',
                'original_size': original_size,
                'compressed_size': original_size,
                'original_size_str': format_file_size(original_size),
                'compressed_size_str': format_file_size(original_size),
                'ratio': 0,
                'no_gain': True,
                'task_id': task_id
            })

        return jsonify({
            'status': 'ok',
            'message': f'压缩完成，减小了 {ratio:.1f}%',
            'original_size': original_size,
            'compressed_size': compressed_size,
            'original_size_str': format_file_size(original_size),
            'compressed_size_str': format_file_size(compressed_size),
            'ratio': round(ratio, 1),
            'task_id': task_id
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'压缩失败: {str(e)}'})


@pdfcompress_bp.route('/pdfcompress/download')
def download():
    task_id = request.args.get('task_id', '')
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效任务'})
    output_path = os.path.join(task_folder, 'output.pdf')
    if not os.path.exists(output_path):
        return jsonify({'status': 'error', 'message': '文件不存在'})
    try:
        with open(os.path.join(task_folder, 'orig_name.txt'), 'r') as f:
            orig_name = f.read().strip()
        download_name = f'{orig_name}_compressed.pdf'
    except Exception:
        download_name = 'compressed.pdf'
    return send_file(output_path, as_attachment=True, download_name=download_name)
