"""PDF Split routes."""
import os
import json
import base64
import fitz
from flask import Blueprint, request, jsonify, send_file
from utils import create_task, get_task_folder, parse_page_ranges, format_file_size, send_zip_from_dir

pdfsplit_bp = Blueprint('pdfsplit', __name__)


@pdfsplit_bp.route('/pdfsplit/upload', methods=['POST'])
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

    orig_name = os.path.splitext(uploaded_file.filename)[0]
    with open(os.path.join(task_folder, 'orig_name.txt'), 'w') as f:
        f.write(orig_name)

    try:
        doc = fitz.open(pdf_path)
        page_count = doc.page_count

        # Render thumbnails at 72 DPI as JPEG base64
        thumbnails = []
        thumb_dir = os.path.join(task_folder, 'thumbs')
        os.makedirs(thumb_dir, exist_ok=True)

        for page_num in range(page_count):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))  # 72 DPI
            tn_path = os.path.join(thumb_dir, f'p{page_num + 1}.jpg')
            pix.save(tn_path, jpg_quality=60)

        doc.close()

        file_size = os.path.getsize(pdf_path)
        return jsonify({
            'status': 'ok',
            'task_id': task_id,
            'page_count': page_count,
            'filename': uploaded_file.filename,
            'file_size_str': format_file_size(file_size)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'PDF文件无法打开: {str(e)}'})


@pdfsplit_bp.route('/pdfsplit/thumb/<task_id>/<int:page_num>')
def thumb(task_id, page_num):
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效任务'})
    tn_path = os.path.join(task_folder, 'thumbs', f'p{page_num}.jpg')
    if not os.path.exists(tn_path):
        return jsonify({'status': 'error', 'message': '缩略图不存在'})
    return send_file(tn_path, mimetype='image/jpeg')


@pdfsplit_bp.route('/pdfsplit/split', methods=['POST'])
def split():
    data = request.get_json(silent=True) or request.form
    task_id = data.get('task_id', '')
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效任务'})

    pdf_path = os.path.join(task_folder, 'input.pdf')
    if not os.path.exists(pdf_path):
        return jsonify({'status': 'error', 'message': '请先上传PDF文件'})

    mode = data.get('mode', 'every_n')
    output_dir = os.path.join(task_folder, 'split_out')
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(task_folder, 'orig_name.txt'), 'r') as f:
        orig_name = f.read().strip()

    try:
        src_doc = fitz.open(pdf_path)
        total_pages = src_doc.page_count

        groups = []  # list of (filename, [page_indices])

        if mode == 'ranges':
            range_str = data.get('ranges', '')
            # Split by spaces: "1-3 4-6 7"
            range_parts = [p.strip() for p in range_str.replace(',', ' ').split() if p.strip()]
            if not range_parts:
                src_doc.close()
                return jsonify({'status': 'error', 'message': '请输入页码范围'})
            for idx, part in enumerate(range_parts):
                pages, err = parse_page_ranges(part, total_pages)
                if err:
                    src_doc.close()
                    return jsonify({'status': 'error', 'message': err})
                groups.append((f'{orig_name}_part{idx + 1}.pdf', pages))

        elif mode == 'selected':
            pages_str = data.get('pages', '')
            try:
                pages = sorted([int(p) for p in pages_str.split(',') if p.strip()])
            except ValueError:
                src_doc.close()
                return jsonify({'status': 'error', 'message': '页面选择无效'})
            if not pages:
                src_doc.close()
                return jsonify({'status': 'error', 'message': '请先选择页面'})
            # Convert from 1-indexed to 0-indexed
            pages_0 = [p - 1 for p in pages if 1 <= p <= total_pages]
            groups.append((f'{orig_name}_selected.pdf', pages_0))

        elif mode == 'every_n':
            n = int(data.get('n', 1))
            if n < 1:
                src_doc.close()
                return jsonify({'status': 'error', 'message': 'N必须大于0'})
            part_idx = 0
            for start in range(0, total_pages, n):
                part_idx += 1
                end = min(start + n, total_pages)
                pages = list(range(start, end))
                groups.append((f'{orig_name}_part{part_idx}.pdf', pages))
        else:
            src_doc.close()
            return jsonify({'status': 'error', 'message': '无效的拆分模式'})

        result_files = []
        for fname, pages in groups:
            out_path = os.path.join(output_dir, fname)
            new_doc = fitz.open()
            for p in pages:
                new_doc.insert_pdf(src_doc, from_page=p, to_page=p)
            new_doc.save(out_path, garbage=4, deflate=True)
            new_doc.close()
            result_files.append({
                'filename': fname,
                'pages': len(pages),
                'page_range': f'{pages[0] + 1}-{pages[-1] + 1}'
            })

        src_doc.close()

        with open(os.path.join(task_folder, 'split_meta.json'), 'w') as f:
            json.dump({'files': [r['filename'] for r in result_files], 'orig_name': orig_name}, f)

        return jsonify({
            'status': 'ok',
            'message': f'拆分完成，生成 {len(result_files)} 个文件',
            'files': result_files,
            'task_id': task_id
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'拆分失败: {str(e)}'})


@pdfsplit_bp.route('/pdfsplit/download_part')
def download_part():
    task_id = request.args.get('task_id', '')
    filename = request.args.get('file', '')
    task_folder = get_task_folder(task_id)
    if not task_folder or not filename:
        return jsonify({'status': 'error', 'message': '参数错误'})
    safe_name = os.path.basename(filename)
    fpath = os.path.join(task_folder, 'split_out', safe_name)
    if not os.path.exists(fpath):
        return jsonify({'status': 'error', 'message': '文件不存在'})
    return send_file(fpath, as_attachment=True, download_name=safe_name)


@pdfsplit_bp.route('/pdfsplit/download_zip')
def download_zip():
    task_id = request.args.get('task_id', '')
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效任务'})
    meta_path = os.path.join(task_folder, 'split_meta.json')
    if not os.path.exists(meta_path):
        return jsonify({'status': 'error', 'message': '请先完成拆分'})
    with open(meta_path) as f:
        meta = json.load(f)
    with open(os.path.join(task_folder, 'orig_name.txt'), 'r') as f:
        orig_name = f.read().strip()
    zip_name = f'{orig_name}_split.zip'
    return send_zip_from_dir(
        os.path.join(task_folder, 'split_out'),
        meta['files'], zip_name
    )
