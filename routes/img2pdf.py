"""Image to PDF conversion routes."""
import os
import json
import fitz
from PIL import Image
from flask import Blueprint, request, jsonify, send_file
from utils import create_task, get_task_folder

img2pdf_bp = Blueprint('img2pdf', __name__)

A4_PORTRAIT_W = 595
A4_PORTRAIT_H = 842
A4_LANDSCAPE_W = 842
A4_LANDSCAPE_H = 595

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def _pil_to_fitz_size(img_path):
    """Get image dimensions in points (72 DPI) using PIL."""
    img = Image.open(img_path)
    w, h = img.size
    img.close()
    return w, h


@img2pdf_bp.route('/img2pdf/upload', methods=['POST'])
def upload():
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'status': 'error', 'message': '请选择图片文件'})

    task_id = create_task()
    task_folder = get_task_folder(task_id)
    img_dir = os.path.join(task_folder, 'images')
    os.makedirs(img_dir, exist_ok=True)

    file_info = []
    saved_names = []
    for i, f in enumerate(files):
        if not f.filename:
            continue
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue
        # Save with index prefix to preserve order
        safe_name = f'{i:04d}_{f.filename}'
        fpath = os.path.join(img_dir, safe_name)
        f.save(fpath)
        try:
            img = Image.open(fpath)
            w, h = img.size
            img.close()
            file_info.append({
                'name': f.filename,
                'saved_name': safe_name,
                'width': w,
                'height': h,
                'size': os.path.getsize(fpath)
            })
            saved_names.append(safe_name)
        except Exception:
            os.remove(fpath)

    if not file_info:
        return jsonify({'status': 'error', 'message': '未找到有效的图片文件，请上传 PNG/JPG/WebP/BMP 格式'})

    with open(os.path.join(task_folder, 'img_order.json'), 'w') as f:
        json.dump(saved_names, f)
    with open(os.path.join(task_folder, 'img_info.json'), 'w') as f:
        json.dump(file_info, f)

    return jsonify({
        'status': 'ok',
        'task_id': task_id,
        'files': file_info,
        'count': len(file_info)
    })


@img2pdf_bp.route('/img2pdf/reorder', methods=['POST'])
def reorder():
    data = request.get_json(silent=True) or {}
    task_id = data.get('task_id', '')
    new_order = data.get('order', [])
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效任务'})

    order_path = os.path.join(task_folder, 'img_order.json')
    if not os.path.exists(order_path):
        return jsonify({'status': 'error', 'message': '请先上传图片'})

    # Validate that all files in new_order exist
    img_dir = os.path.join(task_folder, 'images')
    for name in new_order:
        if not os.path.exists(os.path.join(img_dir, name)):
            return jsonify({'status': 'error', 'message': f'文件 {name} 不存在'})

    with open(order_path, 'w') as f:
        json.dump(new_order, f)

    return jsonify({'status': 'ok'})


@img2pdf_bp.route('/img2pdf/thumb/<task_id>/<filename>')
def thumb(task_id, filename):
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效任务'})
    # Prevent path traversal
    safe_name = os.path.basename(filename)
    fpath = os.path.join(task_folder, 'images', safe_name)
    if not os.path.exists(fpath):
        return jsonify({'status': 'error', 'message': '文件不存在'})
    return send_file(fpath)


@img2pdf_bp.route('/img2pdf/convert', methods=['POST'])
def convert():
    data = request.get_json(silent=True) or request.form
    task_id = data.get('task_id', '')
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效的任务，请重新上传'})

    order_path = os.path.join(task_folder, 'img_order.json')
    img_dir = os.path.join(task_folder, 'images')
    if not os.path.exists(order_path):
        return jsonify({'status': 'error', 'message': '请先上传图片'})

    with open(order_path) as f:
        img_order = json.load(f)

    layout = data.get('layout', 'original')
    auto_rotate = data.get('auto_rotate', False)
    if isinstance(auto_rotate, str):
        auto_rotate = auto_rotate.lower() in ('true', '1', 'yes')

    output_path = os.path.join(task_folder, 'output.pdf')

    try:
        doc = fitz.open()

        for saved_name in img_order:
            img_path = os.path.join(img_dir, saved_name)
            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path)
            img_w, img_h = img.size
            img.close()

            if layout == 'original':
                # Original size at 72 DPI assumption — if image was designed for 72 DPI this is 1:1
                # Use pixel dims directly (treating pixels as points at 72 DPI)
                page_w = img_w * 72 / 96.0  # assume 96 DPI screen images
                page_h = img_h * 72 / 96.0
                rect = fitz.Rect(0, 0, page_w, page_h)
                page = doc.new_page(width=page_w, height=page_h)
                page.insert_image(rect, filename=img_path)

            elif layout == 'a4_portrait':
                pw, ph = A4_PORTRAIT_W, A4_PORTRAIT_H
                if auto_rotate and img_w > img_h:
                    pw, ph = A4_LANDSCAPE_W, A4_LANDSCAPE_H
                page = doc.new_page(width=pw, height=ph)
                scale = min((pw - 40) / img_w, (ph - 40) / img_h)
                draw_w = img_w * scale
                draw_h = img_h * scale
                x0 = (pw - draw_w) / 2
                y0 = (ph - draw_h) / 2
                rect = fitz.Rect(x0, y0, x0 + draw_w, y0 + draw_h)
                page.insert_image(rect, filename=img_path)

            elif layout == 'a4_landscape':
                pw, ph = A4_LANDSCAPE_W, A4_LANDSCAPE_H
                if auto_rotate and img_h > img_w:
                    pw, ph = A4_PORTRAIT_W, A4_PORTRAIT_H
                page = doc.new_page(width=pw, height=ph)
                scale = min((pw - 40) / img_w, (ph - 40) / img_h)
                draw_w = img_w * scale
                draw_h = img_h * scale
                x0 = (pw - draw_w) / 2
                y0 = (ph - draw_h) / 2
                rect = fitz.Rect(x0, y0, x0 + draw_w, y0 + draw_h)
                page.insert_image(rect, filename=img_path)

        doc.save(output_path, garbage=4, deflate=True)
        doc.close()

        file_size = os.path.getsize(output_path)
        from utils import format_file_size
        return jsonify({
            'status': 'ok',
            'message': f'PDF生成成功，共 {len(img_order)} 页',
            'page_count': len(img_order),
            'file_size': file_size,
            'file_size_str': format_file_size(file_size),
            'task_id': task_id
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'生成PDF失败: {str(e)}'})


@img2pdf_bp.route('/img2pdf/download')
def download():
    task_id = request.args.get('task_id', '')
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效任务'})
    output_path = os.path.join(task_folder, 'output.pdf')
    if not os.path.exists(output_path):
        return jsonify({'status': 'error', 'message': '文件不存在，请先转换'})
    return send_file(output_path, as_attachment=True, download_name='converted_images.pdf')
