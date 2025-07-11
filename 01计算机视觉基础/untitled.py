import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 从 model_definition.py 中导入模型类
from model_definition import Net_with_dropout_and_batchnorm

# 类别名称（必须与训练时一致）
class_names = ['cloudy', 'rainy', 'sunny', 'snowy']

# 实例化模型
model = Net_with_dropout_and_batchnorm()

# 加载权重
model.load_state_dict(torch.load('./models/model_weights_NetwithBNandDO.pth'))
model.eval()  # 设置为评估模式

# 图像预处理（必须和训练时一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),         # 假设训练时使用了 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 标准化
])

# 读取图像
image_path = './test_images/test_weather.jpg'
image = Image.open(image_path).convert('RGB')

# 预处理 + 添加 batch 维度 (1, C, H, W)
image_tensor = transform(image).unsqueeze(0)

# 推理
with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted_idx = torch.max(outputs, 1)
    predicted_class = class_names[predicted_idx.item()]

# 可视化结果
plt.imshow(image)
plt.title(f"Predicted: {predicted_class}")
plt.axis('off')
plt.show()

print(f"预测天气为: {predicted_class}")