from flask import Flask, render_template, request, send_file, jsonify
import os
import re
import shutil
import uuid
import time
import cv2
import numpy as np
import fitz
from PIL import Image

TASK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tasks')

CONVERT_DPI = 150

app = Flask(__name__)


def remove_watermark_gray(image, sensitivity=50):
    """去除灰色/半透明灰色水印"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image)

    diff_rg = np.abs(r.astype(np.int16) - g.astype(np.int16))
    diff_rb = np.abs(r.astype(np.int16) - b.astype(np.int16))
    diff_gb = np.abs(g.astype(np.int16) - b.astype(np.int16))
    max_diff = np.maximum(np.maximum(diff_rg, diff_rb), diff_gb)

    s_channel = hsv[:, :, 1]

    # 灰色判定阈值随灵敏度变化
    gray_threshold = 15 + sensitivity * 2 // 5
    sat_threshold = 12 + sensitivity * 2 // 5
    is_gray_pixel = (max_diff < gray_threshold) & (s_channel < sat_threshold)

    # 亮度范围
    brightness_low = 160 + (100 - sensitivity) * 0.8
    is_wm_brightness = (gray >= brightness_low) & (gray < 255)

    # 保护深色内容
    is_dark = gray < 130

    mask = is_gray_pixel & is_wm_brightness & ~is_dark

    # 边缘保护
    edges = cv2.Canny(gray, 80, 200)
    edge_dilate = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    bright_near_edge = gray > 200
    mask = mask & ~(edge_dilate > 0) | (mask & bright_near_edge)

    mask_uint8 = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

    result = image.copy()
    result[mask_uint8 == 255] = [255, 255, 255]
    return result


def remove_watermark_color(image, sensitivity=50):
    """去除彩色水印"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    brightness_low = 160 + (100 - sensitivity)
    is_bright = (gray >= brightness_low) & (gray < 255)
    has_low_sat = s_channel < (30 + sensitivity // 5)
    not_dark = v_channel > 100

    mask = is_bright & has_low_sat & not_dark

    edges = cv2.Canny(gray, 80, 200)
    edge_dilate = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    dark_near_edge = gray < 160
    mask[edge_dilate > 0 & dark_near_edge] = False

    mask_uint8 = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

    result = image.copy()
    result[mask_uint8 == 255] = [255, 255, 255]
    return result


def remove_watermark_dark(image, sensitivity=50):
    """去除深色水印"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image)

    diff_rg = np.abs(r.astype(np.int16) - g.astype(np.int16))
    diff_rb = np.abs(r.astype(np.int16) - b.astype(np.int16))
    diff_gb = np.abs(g.astype(np.int16) - b.astype(np.int16))
    max_diff = np.maximum(np.maximum(diff_rg, diff_rb), diff_gb)

    s_channel = hsv[:, :, 1]

    gray_threshold = 20 + sensitivity // 10
    sat_threshold = 15 + sensitivity // 8
    is_gray_pixel = (max_diff < gray_threshold) & (s_channel < sat_threshold)

    dark_low = 80 + (100 - sensitivity)
    dark_high = 200
    is_dark_watermark = (gray >= dark_low) & (gray < dark_high)

    # 文字保护
    edges = cv2.Canny(gray, 50, 150)
    edge_dilate = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    is_text = (gray < 120) & (edge_dilate > 0)
    text_dilate = cv2.dilate(is_text.astype(np.uint8) * 255,
                             np.ones((5, 5), np.uint8), iterations=2)

    mask = is_gray_pixel & is_dark_watermark & (text_dilate == 0)

    mask_uint8 = mask.astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )
    min_area = 50 + sensitivity * 5
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask_uint8[labels == i] = 0

    result = image.copy()
    result[mask_uint8 == 255] = [255, 255, 255]
    return result


def remove_watermark_auto(image, sensitivity=50):
    """自动检测并去除水印"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image)

    diff_rg = np.abs(r.astype(np.int16) - g.astype(np.int16))
    diff_rb = np.abs(r.astype(np.int16) - b.astype(np.int16))
    diff_gb = np.abs(g.astype(np.int16) - b.astype(np.int16))
    max_diff = np.maximum(np.maximum(diff_rg, diff_rb), diff_gb)

    s_channel = hsv[:, :, 1]

    # 灰色阈值随灵敏度增大而放宽
    gray_threshold = 15 + sensitivity * 3 // 5
    sat_threshold = 12 + sensitivity * 3 // 5
    is_gray_pixel = (max_diff < gray_threshold) & (s_channel < sat_threshold)

    # 亮度范围
    brightness_low = 120 + (100 - sensitivity)
    is_wm_brightness = (gray >= brightness_low) & (gray < 255)

    # 保护深色文字
    is_dark = gray < max(100, brightness_low - 40)

    # 基础掩码
    mask = is_gray_pixel & is_wm_brightness & ~is_dark

    # 边缘保护：在边缘附近更保守
    edges = cv2.Canny(gray, 50, 150)
    edge_dilate = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edge_mask = edge_dilate > 0
    bright_near_edge = gray > 200
    mask = mask & ~(edge_mask & ~bright_near_edge)

    # 形态学清理
    mask_uint8 = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

    result = image.copy()
    result[mask_uint8 == 255] = [255, 255, 255]
    return result


def _try_native_remove(pdf_path, output_path):
    """尝试在PDF原生层面去除水印，成功返回True"""
    doc = fitz.open(pdf_path)
    modified = False

    for page_num in range(doc.page_count):
        page = doc[page_num]

        # 移除水印注释
        annot = page.first_annot
        while annot:
            next_annot = annot.next
            annot_type = annot.type[0] if annot.type else 0
            if annot_type in (1, 2, 3):
                page.delete_annot(annot)
                modified = True
            annot = next_annot

        # 重写content stream移除水印
        xrefs = page.get_contents()
        for xref in xrefs:
            stream = doc.xref_stream(xref)
            if stream:
                text = stream.decode('latin-1', errors='replace')
                new_text = _remove_watermark_from_stream(text)
                if new_text != text:
                    doc.update_stream(xref, new_text.encode('latin-1'))
                    modified = True

    if modified:
        doc.save(output_path, garbage=0, deflate=True)
        doc.close()
        return True

    doc.close()
    return False


def _remove_watermark_from_stream(stream_text):
    """从PDF content stream中移除水印指令"""
    lines = stream_text.split('\n')
    result_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 检测 q...Q 块中的水印
        if line == 'q':
            block_end = _find_matching_q(lines, i)
            if block_end >= 0:
                block = '\n'.join(lines[i:block_end + 1])
                if _is_watermark_block(block, stream_text):
                    i = block_end + 1
                    continue

        # 检测独立的旋转文字行
        if _is_rotated_text_line(line):
            i += 1
            continue

        result_lines.append(lines[i])
        i += 1

    return '\n'.join(result_lines)


def _find_matching_q(lines, start):
    depth = 0
    for i in range(start, len(lines)):
        line = lines[i].strip()
        if line == 'q':
            depth += 1
        elif line == 'Q':
            depth -= 1
            if depth == 0:
                return i
    return -1


def _is_watermark_block(block_text, full_stream):
    has_gs = 'gs' in block_text
    has_transparency = '/ca' in full_stream or '/CA' in full_stream

    # 检查水印关键词
    lower = block_text.lower()
    keywords = ['confidential', 'watermark', 'draft', 'sample',
                'do not copy', '内部', '机密', '样本', '草稿']
    has_keyword = any(kw in lower for kw in keywords)

    # 检查旋转矩阵 (cm 操作符)
    has_rotation = False
    cm_pattern = r'([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+cm'
    for match in re.finditer(cm_pattern, block_text):
        a, b, c, d = (float(match.group(k)) for k in range(1, 5))
        if abs(abs(a) - abs(d)) < 0.01 and abs(abs(b) - abs(c)) < 0.01 and abs(b) > 0.5:
            has_rotation = True

    # 匹配条件：关键词 OR (旋转 + 透明度) OR (关键词在完整流中 + 旋转)
    if has_keyword and has_rotation:
        return True
    if has_keyword and has_gs and has_transparency:
        return True
    if has_rotation and has_gs and has_transparency:
        return True
    if has_keyword:
        return True

    return False


def _is_rotated_text_line(line):
    if 'cm' not in line:
        return False
    parts = line.split()
    if len(parts) >= 7 and parts[-1] == 'cm':
        try:
            a, b, c, d = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            if abs(abs(a) - abs(d)) < 0.01 and abs(abs(b) - abs(c)) < 0.01 and abs(b) > 0.5:
                return True
        except (ValueError, IndexError):
            pass
    return False


def remove_watermark_pipeline(pdf_path, output_path, sensitivity=50, mode='auto'):
    """水印去除流水线"""
    # 第一步：尝试PDF原生去除
    native_output = output_path + '.native.pdf'
    native_success = _try_native_remove(pdf_path, native_output)
    work_pdf = native_output if native_success else pdf_path

    # 第二步：图片级别处理
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_wm_temp')
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(work_pdf)
    scale = CONVERT_DPI / 72
    image_paths = []

    choose_func = {
        'gray': remove_watermark_gray,
        'color': remove_watermark_color,
        'dark': remove_watermark_dark,
        'auto': remove_watermark_auto,
    }.get(mode, remove_watermark_auto)

    for page_num in range(doc.page_count):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        img_array = np.frombuffer(pix.tobytes('png'), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        img = choose_func(img, sensitivity)

        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        success, encoded = cv2.imencode('.png', img)
        if success:
            with open(image_path, 'wb') as f:
                f.write(encoded.tobytes())
        image_paths.append(image_path)

    doc.close()

    # 第三步：合并为PDF
    images_to_pdf(image_paths, output_path)

    # 清理
    shutil.rmtree(output_folder, ignore_errors=True)
    if native_success and os.path.exists(native_output):
        os.remove(native_output)


def images_to_pdf(image_paths, output_path):
    doc = fitz.open()

    for image_path in image_paths:
        img = Image.open(image_path)
        width, height = img.size

        page_width = width * 72 / CONVERT_DPI
        page_height = height * 72 / CONVERT_DPI

        page = doc.new_page(width=page_width, height=page_height)
        rect = fitz.Rect(0, 0, page_width, page_height)
        page.insert_image(rect, filename=image_path)

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    _cleanup_old_tasks()
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        task_id = uuid.uuid4().hex
        task_folder = os.path.join(TASK_DIR, task_id)
        os.makedirs(task_folder, exist_ok=True)
        pdf_path = os.path.join(task_folder, 'input.pdf')
        uploaded_file.save(pdf_path)
        return jsonify({'status': 'ok', 'message': '文件上传成功', 'task_id': task_id})


@app.route('/remove_watermark', methods=['GET', 'POST'])
def remove_watermark_route():
    task_id = request.args.get('task_id', '')
    if not task_id or not os.path.isdir(os.path.join(TASK_DIR, task_id)):
        return jsonify({'status': 'error', 'message': '无效的任务，请重新上传文件'})

    sensitivity = request.args.get('sensitivity', 50, type=int)
    sensitivity = max(1, min(100, sensitivity))
    mode = request.args.get('mode', 'auto')

    if mode not in ('gray', 'color', 'dark', 'auto'):
        mode = 'auto'

    task_folder = os.path.join(TASK_DIR, task_id)
    pdf_path = os.path.join(task_folder, 'input.pdf')
    if not os.path.exists(pdf_path):
        return jsonify({'status': 'error', 'message': '请先上传PDF文件'})

    output_pdf_path = os.path.join(task_folder, 'output.pdf')
    remove_watermark_pipeline(pdf_path, output_pdf_path, sensitivity, mode)
    return jsonify({
        'status': 'ok',
        'message': f'水印去除成功 (模式: {mode}, 灵敏度: {sensitivity})',
        'task_id': task_id
    })


@app.route('/download')
def download():
    task_id = request.args.get('task_id', '')
    if not task_id:
        return jsonify({'status': 'error', 'message': '缺少任务ID'})
    output_path = os.path.join(TASK_DIR, task_id, 'output.pdf')
    if not os.path.exists(output_path):
        return jsonify({'status': 'error', 'message': '文件不存在'})
    return send_file(output_path, as_attachment=True)


def _cleanup_old_tasks(max_age_hours=1):
    """清理超过指定时间的旧任务"""
    if not os.path.isdir(TASK_DIR):
        return
    now = time.time()
    for name in os.listdir(TASK_DIR):
        path = os.path.join(TASK_DIR, name)
        if os.path.isdir(path) and now - os.path.getmtime(path) > max_age_hours * 3600:
            shutil.rmtree(path, ignore_errors=True)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
