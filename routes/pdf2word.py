"""PDF to Word conversion routes (pdf2docx)."""
import os
import json
from flask import Blueprint, request, jsonify, send_file
from utils import create_task, get_task_folder, parse_page_ranges, format_file_size

pdf2word_bp = Blueprint('pdf2word', __name__)


@pdf2word_bp.route('/pdf2word/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    if not f or not f.filename:
        return jsonify({'status': 'error', 'message': '请选择PDF文件'})
    if not f.filename.lower().endswith('.pdf'):
        return jsonify({'status': 'error', 'message': '仅支持 .pdf 格式'})

    task_id = create_task()
    task_folder = get_task_folder(task_id)
    saved = os.path.join(task_folder, 'input.pdf')
    f.save(saved)

    try:
        import fitz
        doc = fitz.open(saved)
        page_count = doc.page_count
        doc.close()
    except Exception:
        os.remove(saved)
        return jsonify({'status': 'error', 'message': '无法解析该PDF文件'})

    with open(os.path.join(task_folder, 'meta.json'), 'w', encoding='utf-8') as fp:
        json.dump({'filename': f.filename}, fp, ensure_ascii=False)

    size = os.path.getsize(saved)
    return jsonify({
        'status': 'ok',
        'task_id': task_id,
        'filename': f.filename,
        'page_count': page_count,
        'size': size,
        'size_str': format_file_size(size)
    })


@pdf2word_bp.route('/pdf2word/convert', methods=['POST'])
def convert():
    data = request.get_json(silent=True) or {}
    task_id = data.get('task_id', '')
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效任务'})

    src = os.path.join(task_folder, 'input.pdf')
    if not os.path.exists(src):
        return jsonify({'status': 'error', 'message': '未找到上传的PDF，请重新上传'})

    pages = None
    page_range = (data.get('page_range') or 'all').strip()
    if page_range and page_range.lower() != 'all':
        import fitz
        doc = fitz.open(src)
        total = doc.page_count
        doc.close()
        pages, err = parse_page_ranges(page_range, total)
        if err:
            return jsonify({'status': 'error', 'message': err})

    output_path = os.path.join(task_folder, 'output.docx')
    try:
        from pdf2docx import Converter
        cv = Converter(src)
        try:
            if pages:
                cv.convert(output_path, pages=pages)
            else:
                cv.convert(output_path)
        finally:
            cv.close()
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'转换失败: {e}'})

    if not os.path.exists(output_path):
        return jsonify({'status': 'error', 'message': '转换失败：未生成Word文档'})

    size = os.path.getsize(output_path)
    return jsonify({
        'status': 'ok',
        'message': '转换成功！文字与段落已转为可编辑内容',
        'file_size': size,
        'file_size_str': format_file_size(size),
        'task_id': task_id
    })


@pdf2word_bp.route('/pdf2word/download')
def download():
    task_id = request.args.get('task_id', '')
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效任务'})
    output_path = os.path.join(task_folder, 'output.docx')
    if not os.path.exists(output_path):
        return jsonify({'status': 'error', 'message': '文件不存在，请先转换'})

    download_name = 'converted.docx'
    meta_path = os.path.join(task_folder, 'meta.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding='utf-8') as fp:
                name = json.load(fp).get('filename', '')
            if name:
                download_name = os.path.splitext(os.path.basename(name))[0] + '.docx'
        except Exception:
            pass

    return send_file(output_path, as_attachment=True, download_name=download_name)
