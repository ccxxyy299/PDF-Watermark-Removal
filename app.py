from flask import Flask, render_template, request, send_file, jsonify
import os
import cv2
import numpy as np
import fitz
from PIL import Image
import tempfile

# DPI 设置
CONVERT_DPI = 400

app = Flask(__name__)


def remove_watermark(image_path, sensitivity=50):
    """
    去除半透明水印 - 通用版本
    sensitivity: 灵敏度 1-100
    """
    img = cv2.imread(image_path)
    if img is None:
        return

    # 转换色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l_channel = lab[:, :, 0]
    s_channel = hsv[:, :, 1]

    # ===== 基于颜色的水印检测 =====
    # 水印特征: 高亮度 + 极低饱和度
    lower_l = 205 - sensitivity // 5
    upper_l = 250
    sat_max = 5 + sensitivity // 20

    # 亮度掩码
    brightness_mask = cv2.inRange(l_channel, lower_l, upper_l)
    # 饱和度掩码（水印是灰色的）
    sat_mask = (s_channel < sat_max).astype(np.uint8) * 255
    # 保护有颜色的内容
    color_protection = (s_channel > 20).astype(np.uint8) * 255

    # 组合：高亮度 + 低饱和度 + 非彩色内容
    final_mask = cv2.bitwise_and(brightness_mask, sat_mask)
    final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(color_protection))

    # ===== 应用掩码 =====
    img[final_mask == 255] = [255, 255, 255]

    cv2.imwrite(image_path, img)


def pdf_to_images(pdf_path, output_folder, sensitivity=50):
    """将 PDF 转换为图片"""
    images = []
    doc = fitz.open(pdf_path)
    scale = CONVERT_DPI / 72

    for page_num in range(doc.page_count):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        # 使用无损 PNG 保存
        pix.save(image_path)
        images.append(image_path)
        remove_watermark(image_path, sensitivity)
    return images


def images_to_pdf(image_paths, output_path):
    """将图片合并为 PDF，保持原始质量"""
    doc = fitz.open()

    for image_path in image_paths:
        img = Image.open(image_path)
        width, height = img.size

        # 根据 DPI 计算页面尺寸（英寸转点）
        page_width = width * 72 / CONVERT_DPI
        page_height = height * 72 / CONVERT_DPI

        # 创建页面
        page = doc.new_page(width=page_width, height=page_height)

        # 直接插入图片，保持原始质量
        rect = fitz.Rect(0, 0, page_width, page_height)
        page.insert_image(rect, filename=image_path)

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        pdf_path = 'uploads/uploaded_file.pdf'
        uploaded_file.save(pdf_path)
        return render_template('index.html', message='文件上传成功')


@app.route('/remove_watermark', methods=['GET'])
def remove_watermark_route():
    # 获取灵敏度参数，默认 50
    sensitivity = request.args.get('sensitivity', 50, type=int)
    sensitivity = max(1, min(100, sensitivity))  # 限制在 1-100

    pdf_path = 'uploads/uploaded_file.pdf'
    output_folder = 'output_images'
    os.makedirs(output_folder, exist_ok=True)  # 创建输出目录（如果不存在）
    image_paths = pdf_to_images(pdf_path, output_folder, sensitivity)
    output_pdf_path = 'output_file.pdf'
    images_to_pdf(image_paths, output_pdf_path)
    return render_template('index.html', message=f'水印去除成功 (灵敏度: {sensitivity})')


@app.route('/download')
def download():
    output_pdf_path = 'output_file.pdf'
    return send_file(output_pdf_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
