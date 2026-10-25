"""Word to PDF conversion routes (via local MS Word COM, high fidelity)."""
import os
import json
from flask import Blueprint, request, jsonify, send_file
from utils import create_task, get_task_folder, format_file_size

word2pdf_bp = Blueprint('word2pdf', __name__)


@word2pdf_bp.route('/word2pdf/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    if not f or not f.filename:
        return jsonify({'status': 'error', 'message': '请选择Word文档'})

    name = f.filename
    ext = os.path.splitext(name)[1].lower()
    if ext not in ('.doc', '.docx'):
        return jsonify({'status': 'error', 'message': '仅支持 .doc / .docx 格式的Word文档'})

    task_id = create_task()
    task_folder = get_task_folder(task_id)
    saved = os.path.join(task_folder, 'input' + ext)
    f.save(saved)

    with open(os.path.join(task_folder, 'meta.json'), 'w', encoding='utf-8') as fp:
        json.dump({'filename': name}, fp, ensure_ascii=False)

    size = os.path.getsize(saved)
    return jsonify({
        'status': 'ok',
        'task_id': task_id,
        'filename': name,
        'size': size,
        'size_str': format_file_size(size)
    })


@word2pdf_bp.route('/word2pdf/convert', methods=['POST'])
def convert():
    data = request.get_json(silent=True) or {}
    task_id = data.get('task_id', '')
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效任务'})

    src = None
    for ext in ('.docx', '.doc'):
        p = os.path.join(task_folder, 'input' + ext)
        if os.path.exists(p):
            src = p
            break
    if not src:
        return jsonify({'status': 'error', 'message': '未找到上传的Word文档，请重新上传'})

    output_path = os.path.join(task_folder, 'output.pdf')
    try:
        from docx2pdf import convert
        convert(src, output_path)
    except Exception as e:
        msg = str(e)
        if 'Word' in msg or '0x800A03EC' in msg or 'COM' in msg.upper():
            msg = '调用本机 Word 转换失败，请确认已安装 Microsoft Word 且未弹出安全对话框'
        return jsonify({'status': 'error', 'message': f'转换失败: {msg}'})

    if not os.path.exists(output_path):
        return jsonify({'status': 'error', 'message': '转换失败：未生成PDF文件'})

    size = os.path.getsize(output_path)
    return jsonify({
        'status': 'ok',
        'message': '转换成功！Word排版已完整保留（基于本机 Word 渲染）',
        'file_size': size,
        'file_size_str': format_file_size(size),
        'task_id': task_id
    })


@word2pdf_bp.route('/word2pdf/download')
def download():
    task_id = request.args.get('task_id', '')
    task_folder = get_task_folder(task_id)
    if not task_folder:
        return jsonify({'status': 'error', 'message': '无效任务'})
    output_path = os.path.join(task_folder, 'output.pdf')
    if not os.path.exists(output_path):
        return jsonify({'status': 'error', 'message': '文件不存在，请先转换'})

    download_name = 'converted.pdf'
    meta_path = os.path.join(task_folder, 'meta.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding='utf-8') as fp:
                name = json.load(fp).get('filename', '')
            if name:
                download_name = os.path.splitext(name)[0] + '.pdf'
        except Exception:
            pass

    return send_file(output_path, as_attachment=True, download_name=download_name)
