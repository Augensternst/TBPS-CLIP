import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

from misc.utils import parse_config, init_distributed_mode, set_seed
from misc.build import load_checkpoint
from text_utils.tokenizer import tokenize
from model.tbps_model import clip_vitb

# 配置
config_path = 'E:\TBPS-CLIP\config\config.yaml'  # 替换为实际的配置文件路径
model_path = 'E:\TBPS-CLIP\checkpoint'  # 替换为实际的检查点路径
image_dir = 'imagelib'  # 图库路径
text_input = input("Enter a description to search for images: ")

# 加载配置
config = parse_config(config_path)

# 初始化分布式模式（如果适用）
#init_distributed_mode(config)

# 设置随机种子
set_seed(config)

# 初始化模型
model = clip_vitb(config)
model, _ = load_checkpoint(model, config)
model = model.to(config.device)
model.eval()

# 文本处理
text_tokens = tokenize([text_input], context_length=config.experiment.text_length).to(config.device)

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(config.experiment.input_resolution),
    transforms.CenterCrop(config.experiment.input_resolution),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载并处理图像
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
image_tensors = []

for image_file in image_files:
    image = Image.open(image_file).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(config.device)
    image_tensors.append(image_tensor)

image_tensors = torch.cat(image_tensors)

# 获取文本和图像的特征
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    image_features = model.encode_image(image_tensors)

    text_features = F.normalize(text_features, dim=-1)
    image_features = F.normalize(image_features, dim=-1)

# 计算相似度
similarity = text_features @ image_features.T
similarity_scores = similarity.cpu().numpy()[0]

# 获取最相似的图像
top_k = similarity_scores.argsort()[-5:][::-1]

# 显示结果
plt.figure(figsize=(15, 5))
for i, index in enumerate(top_k):
    plt.subplot(1, 5, i+1)
    plt.imshow(Image.open(image_files[index]))
    plt.axis('off')
    plt.title(f"Score: {similarity_scores[index]:.2f}")

plt.show()
