"""PDF Merge routes."""
import os
import json
import fitz
from flask import Blueprint, request, jsonify, send_file
from utils import create_task, get_task_folder, parse_page_ranges, format_file_size

pdfmerge_bp = Blueprint('pdfmerge', __name__)


@pdfmerge_bp.route('/pdfmerge/upload', methods=['POST'])
def upload():
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'status': 'error', 'message': '请选择PDF文件'})

    task_id = create_task()
    task_folder = get_task_folder(task_id)
    pdf_dir = os.path.join(task_folder, 'pdfs')
    os.makedirs(pdf_dir, exist_ok=True)

    file_info = []
    for i, f in enumerate(files):
        if not f.filename:
            continue
        if not f.filename.lower().endswith('.pdf'):
            continue
        safe_name = f'{i:04d}_{f.filename}'
        fpath = os.path.join(pdf_dir, safe_name)
        f.save(fpath)
        try:
            doc = fitz.open(fpath)
            page_count = doc.page_count
            file_size = os.path.getsize(fpath)
            doc.close()
            file_info.append({
                'uid': safe_name,
                'name': f.filename,
                'saved_name': safe_name,
                'page_count': page_count,
                'size': file_size,
                'size_str': format_file_size(file_size),
                'page_range': 'all'
            })
        except Exception:
            os.remove(fpath)

    if not file_info:
        return jsonify({'status': 'error', 'message': '未找到有效的PDF文件'})

    with open(os.path.join(task_folder, 'merge_order.json'), 'w') as f:
        json.dump([fi['saved_name'] for fi in file_info], f)
    with open(os.path.join(task_folder, 'merge_info.json'), 'w') as f:
        json.dump(file_info, f)

    total_pages = sum(fi['page_count'] for fi in file_info)
    return jsonify({
        'status': 'ok',
        'task_id': task_id,
        'files': file_info,
        'total_pages': total_pages
    })


@pdfmerge_bp.route('/pdfmerge/reorder', methods=['POST'])
def reorder():
    data = request.get_json(silent=True) or {}
    task_id = data.get('task_id', '')
    new_order = data.get('order', [])
    ranges = data.get('ranges', {})
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效任务'})

    pdf_dir = os.path.join(task_folder, 'pdfs')
    for name in new_order:
        if not os.path.exists(os.path.join(pdf_dir, name)):
            return jsonify({'status': 'error', 'message': f'文件不存在: {name}'})

    with open(os.path.join(task_folder, 'merge_order.json'), 'w') as f:
        json.dump(new_order, f)

    # Update page ranges
    info_path = os.path.join(task_folder, 'merge_info.json')
    if os.path.exists(info_path):
        with open(info_path) as f:
            file_info = json.load(f)
        for fi in file_info:
            if fi['saved_name'] in ranges:
                fi['page_range'] = ranges[fi['saved_name']]
        with open(info_path, 'w') as f:
            json.dump(file_info, f)

    # Recalculate total pages estimate
    total = 0
    for fi in file_info:
        if fi['saved_name'] in new_order:
            pr = ranges.get(fi['saved_name'], 'all')
            if pr == 'all' or pr.strip() == '':
                total += fi['page_count']
            else:
                pages, err = parse_page_ranges(pr, fi['page_count'])
                if pages:
                    total += len(pages)
                else:
                    total += fi['page_count']

    return jsonify({'status': 'ok', 'estimated_pages': total})


@pdfmerge_bp.route('/pdfmerge/merge', methods=['POST'])
def merge():
    data = request.get_json(silent=True) or request.form
    task_id = data.get('task_id', '')
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效任务'})

    order_path = os.path.join(task_folder, 'merge_order.json')
    info_path = os.path.join(task_folder, 'merge_info.json')
    if not os.path.exists(order_path) or not os.path.exists(info_path):
        return jsonify({'status': 'error', 'message': '请先上传PDF文件'})

    with open(order_path) as f:
        order = json.load(f)
    with open(info_path) as f:
        file_info = json.load(f)

    # Build lookup
    info_map = {fi['saved_name']: fi for fi in file_info}

    output_path = os.path.join(task_folder, 'output.pdf')
    total_pages_out = 0

    try:
        merged_doc = fitz.open()

        for saved_name in order:
            fpath = os.path.join(task_folder, 'pdfs', saved_name)
            if not os.path.exists(fpath):
                continue

            fi = info_map.get(saved_name, {})
            total_src = fi.get('page_count', 0)
            page_range = fi.get('page_range', 'all')

            src_doc = fitz.open(fpath)
            try:
                if page_range == 'all' or page_range.strip() == '':
                    merged_doc.insert_pdf(src_doc)
                    total_pages_out += src_doc.page_count
                else:
                    pages, err = parse_page_ranges(page_range, src_doc.page_count)
                    if err:
                        src_doc.close()
                        merged_doc.close()
                        return jsonify({'status': 'error', 'message': f'{fi["name"]}: {err}'})
                    # insert_pdf with from_page/to_page only does contiguous ranges
                    # For non-contiguous, insert one page at a time
                    for p in pages:
                        merged_doc.insert_pdf(src_doc, from_page=p, to_page=p)
                        total_pages_out += 1
            finally:
                src_doc.close()

        merged_doc.save(output_path, garbage=4, deflate=True)
        merged_doc.close()

        file_size = os.path.getsize(output_path)
        return jsonify({
            'status': 'ok',
            'message': f'合并完成，共 {total_pages_out} 页',
            'page_count': total_pages_out,
            'file_size': file_size,
            'file_size_str': format_file_size(file_size),
            'task_id': task_id
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'合并失败: {str(e)}'})


@pdfmerge_bp.route('/pdfmerge/download')
def download():
    task_id = request.args.get('task_id', '')
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效任务'})
    output_path = os.path.join(task_folder, 'output.pdf')
    if not os.path.exists(output_path):
        return jsonify({'status': 'error', 'message': '文件不存在，请先合并'})
    return send_file(output_path, as_attachment=True, download_name='merged.pdf')
