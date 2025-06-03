import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import models,transforms
from PIL import Image
from PIL import ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify
from io import BytesIO
import base64
import time

from misc.utils import parse_config, set_seed
from misc.build import load_checkpoint
from text_utils.tokenizer import tokenize
from model.tbps_model import clip_vitb

# Flask 应用
app = Flask(__name__)

# 配置文件路径
config_path = 'config/config.yaml'  # 替换为实际的配置文件路径

# 加载配置
config = parse_config(config_path)

# 设置随机种子
set_seed(config)

# 初始化模型
model = clip_vitb(config)
model, _ = load_checkpoint(model, config)
model = model.to(config.device)
model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(config.experiment.input_resolution),
    transforms.CenterCrop(config.experiment.input_resolution),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 使用 YOLOv3 进行人体检测
# class YOLOv3Detector:
#     def __init__(self, model_path, device='cpu'):
#         self.model = torch.hub.load('ultralytics/yolov3', 'yolov3')
#         self.model.eval()
#         self.device = device
#
#     def detect(self, image):
#         results = self.model(image)
#         boxes = results.xyxy[0].cpu().numpy()  # xyxy format
#         return boxes
#
# # 初始化 YOLOv3 模型
# detector = YOLOv3Detector('ultralytics/yolov3', device=config.device)
#
# def extract_persons(image):
#     # 检测图像中的人体
#     boxes = detector.detect(image)
#     persons = []
#     for box in boxes:
#         if box[5] == 0:  # class 0 corresponds to 'person'
#             box = [int(b) for b in box[:4]]
#             person = image.crop((box[0], box[1], box[2], box[3]))
#             persons.append(person)
#     return persons


# 使用 Faster R-CNN 进行人体检测
class FasterRCNNDetector:
    def __init__(self, device='cpu'):
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model = self.model.to(device)  # 将模型移动到指定设备
        self.model.eval()
        self.device = device

    def detect(self, image):
        transform = transforms.ToTensor()
        image_tensor = transform(image).unsqueeze(0).to(self.device)  # 将输入数据移动到同一设备
        with torch.no_grad():
            detections = self.model(image_tensor)[0]
        return detections

# 初始化 Faster R-CNN 模型
detector = FasterRCNNDetector(device=config.device)

def extract_persons(image):
    # 检测图像中的人体
    detections = detector.detect(image)
    persons = []
    for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        if label == 1 and score > 0.8:  # 1 corresponds to 'person' class
            box = [int(b) for b in box]
            person = image.crop((box[0], box[1], box[2], box[3]))
            persons.append(person)
    return persons


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def video_to_frames(video_path, time_interval=1):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件：{video_path}")
        return [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * time_interval)
    frames = []
    frame_times = []

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 仅保存每间隔 frame_interval 的帧
        if frame_count % frame_interval == 0:
            frames.append(frame)
            frame_times.append(frame_count / fps)

        frame_count += 1

    cap.release()
    return frames, frame_times


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/function')
def render_function():
    return render_template('function.html')



@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()  # 开始时间记录
    print("in")

    # 检查请求中是否有标志来决定使用图片搜索还是文字搜索
    use_image_search = 0

    # 从请求中获取视频文件
    # video_file = request.files['video']
    # video_path = os.path.join("uploads", video_file.filename)
    # video_file.save(video_path)

    video_path = "./static/videos/video.mp4"

    # 抽帧
    frames, frame_times = video_to_frames(video_path, time_interval=10)
    print(frame_times)

    if use_image_search:
        # 从请求中获取搜索的图像Base64编码
        search_images_data = request.json.get('images', [])
        search_images = []

        # 解码Base64编码为PIL图像
        for image_str in search_images_data:
            image = Image.open(BytesIO(base64.b64decode(image_str.split(',')[1]))).convert("RGB")
            search_images.append(image)

        all_image_features = []
        max_similarity_scores = []

        with torch.no_grad():
            # 获取搜索图像的特征
            for search_image in search_images:
                search_image_tensor = preprocess(search_image).unsqueeze(0).to(config.device)
                search_image_features = model.encode_image(search_image_tensor)
                search_image_features = F.normalize(search_image_features, dim=-1)

                for frame_index, (frame, frame_time) in enumerate(zip(frames, frame_times)):
                    persons = extract_persons(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    if persons:
                        person_tensors = [preprocess(person).unsqueeze(0).to(config.device) for person in persons]
                        person_features = [model.encode_image(person_tensor) for person_tensor in person_tensors]
                        person_features = [F.normalize(person_feature, dim=-1) for person_feature in person_features]

                        # 计算相似度
                        similarities = [torch.matmul(search_image_features, person_feature.T).item() for person_feature in person_features]
                        max_similarity_scores.append((max(similarities), frame_index, frame, frame_time))
                    else:
                        max_similarity_scores.append((0, frame_index, frame, frame_time))

        # 获取最相似的图像索引
        max_similarity_scores.sort(key=lambda x: x[0], reverse=True)
        top_k = max_similarity_scores[:5]

        # 转换最相似的图像为 Base64 编码字符串
        top_k_images = [image_to_base64(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) for _, _, frame, _ in top_k]
        top_k_times = [frame_time for _, _, _, frame_time in top_k]
        top_k_indices = [frame_index for _, frame_index, _, _ in top_k]

    else:
        # 保持文字搜索部分不变

        # 从请求中获取文本描述
        print("从请求中获取文本描述")
        text_input = request.json['query']

        # 文本处理
        print("文本处理")
        text_tokens = tokenize([text_input], context_length=config.experiment.text_length).to(config.device)

        # 图像处理
        print("图像处理")
        image_tensors = [
            preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(config.device) for frame
            in frames]

        all_image_features = []
        original_images = []

        # 获取文本和图像的特征
        print("获取文本和图像的特征")
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
            print("进入循环")
            i = 0
            for image_tensor in image_tensors:
                print(f"循环中{i}")
                image_features = model.encode_image(image_tensor)
                image_features = F.normalize(image_features, dim=-1)
                all_image_features.append(image_features)
                original_images.append(image_tensor)
                i += 1

        # 将所有图像特征拼接起来
        print("将所有图像特征拼接起来")
        all_image_features = torch.cat(all_image_features, dim=0)

        # 计算相似度
        print("计算相似度")
        similarity = text_features @ all_image_features.T
        similarity_scores = similarity.cpu().numpy()[0]

        # 获取最相似的图像索引
        top_k = similarity_scores.argsort()[-5:][::-1]

        # 转换最相似的图像为 Base64 编码字符串
        top_k_images = [image_to_base64(Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))) for i in top_k]
        top_k_times = [frame_times[i] for i in top_k]

    # 结束时间记录
    end_time = time.time()

    # 计算并打印耗时
    processing_time = end_time - start_time
    print(f"Processing time: {processing_time:.2f} seconds")

    # 返回结果
    result = {
        # "top_k_indices": top_k_indices,
        "scores": top_k_times,
        "images": top_k_images,
        "times": top_k_times
    }

    return jsonify(result)


if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(host='0.0.0.0', port=5000)