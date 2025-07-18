# 将训练的模型保存为.pt格式详细  

在 PyTorch 中，模型文件的扩展名（如 `.pt`、`.pth` 或 `.bin`）其实只是文件名的约定，并没有强制要求。PyTorch 本身使用 `torch.save()` 函数将模型保存为 **序列化的字节文件**，通常是使用 `.pth` 或 `.pt` 扩展名。

所以，你完全可以将模型保存为 `.pt` 文件，只需在保存时把文件名的后缀改为 `.pt` 即可。

---

### ✅ 保存模型为 `.pt` 文件的方法：

#### 方法 1：保存整个模型
```python
import torch

# 假设 model 是你训练好的模型
torch.save(model, 'model.pt')  # 保存整个模型为 .pt 文件
```

加载模型：
```python
model = torch.load('model.pt')
model.eval()  # 设置为评估模式
```

> ⚠️ 注意：保存整个模型依赖模型的类定义，也就是说加载模型时必须能访问模型的类定义。

---

#### 方法 2：推荐做法：保存模型的状态字典（state_dict）

```python
torch.save(model.state_dict(), 'model.pt')  # 保存模型参数为 .pt 文件
```

加载模型：
```python
model.load_state_dict(torch.load('model.pt'))
model.eval()
```

这种方式更轻量、更推荐，因为你只需要保存和加载模型参数。

---

### 📝 总结：

| 保存方式     | 示例代码                                     | 文件扩展名建议 |
| ------------ | -------------------------------------------- | -------------- |
| 保存整个模型 | `torch.save(model, 'model.pt')`              | `.pt` / `.pth` |
| 保存模型参数 | `torch.save(model.state_dict(), 'model.pt')` | `.pt` / `.pth` |

---

### 💡 小技巧：

你可以使用任意扩展名，比如 `.bin`、`.pth.tar` 等，只要保存和加载时一致即可。

如果你希望文件名统一为 `.pt`，只需在保存时指定 `.pt` 后缀即可。


##  将模型保存为torchscript格式  

### 什么是 TorchScript？

TorchScript 是 PyTorch 提供的一种中间表示（IR），它允许你将 PyTorch 模型转换为独立于 Python 运行时的格式。这使得模型可以在不支持 Python 的环境中运行，比如移动设备、服务器等。此外，TorchScript 也优化了模型执行效率，并且可以通过 JIT (Just-In-Time) 编译器进行进一步的性能提升。

### 如何将 PyTorch 模型转换为 TorchScript 格式？

有两种主要的方法可以实现这一转换：**追踪（Tracing）** 和 **注释脚本模式（Scripting）**。

#### 方法一：通过追踪（Tracing）转换

追踪是通过提供一个具体的输入样例来记录下模型的操作流程。这对于没有控制流的简单模型非常有效。

```python
import torch

# 假设我们有一个简单的模型
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

model = MyModel()

# 创建一个示例输入
example_input = torch.rand(1, 10)

# 使用 torch.jit.trace 来追踪模型
traced_script_module = torch.jit.trace(model, example_input)

# 保存到文件
traced_script_module.save("traced_model.pt")
```

#### 方法二：通过注释脚本模式（Scripting）转换

对于包含复杂控制流的模型，直接使用 `torch.jit.script` 函数可能会更合适。这种方法需要你对模型代码做一些兼容性修改，以便 TorchScript 能够理解。

```python
import torch

# 定义一个带有条件语句的模型
class MyModelWithControlFlow(torch.nn.Module):
    def __init__(self):
        super(MyModelWithControlFlow, self).__init__()

    def forward(self, x):
        if x.sum() > 0:
            return x + 1
        else:
            return x - 1

model = MyModelWithControlFlow()

# 使用 torch.jit.script 直接转换模型
scripted_module = torch.jit.script(model)

# 保存到文件
scripted_module.save("scripted_model.pt")
```

### 总结

- **Tracing** 更适合结构较为固定的模型。
- **Scripting** 更加灵活，适用于有复杂控制流的模型，但可能需要调整模型定义以符合 TorchScript 的语法要求。

两种方法都可以将你的 PyTorch 模型转换为 TorchScript 格式，并将其保存为 `.pt` 文件，从而方便部署到不同的环境。根据你的具体需求选择合适的转换方式。

## 将模型转化为onnx格式并利用tensorRT解析ONNX模型  

### 什么是 TensorRT？

**TensorRT** 是由 NVIDIA 开发的一个高性能深度学习推理（Inference）优化工具。它能够对训练好的模型进行优化，从而在 NVIDIA 的 GPU 上实现更快的推理速度和更低的延迟。TensorRT 支持多种框架训练出来的模型，包括 TensorFlow、PyTorch、ONNX 等，并通过一系列优化技术（如层融合、精度校准等）来提高模型的执行效率。

### 导出为 TensorRT 格式的步骤

要将你的 PyTorch 模型转换为 TensorRT 格式，通常需要经过以下几个步骤：

1. **导出为 ONNX 格式**：首先，你需要将 PyTorch 模型转换为 ONNX 格式。
2. **使用 TensorRT 解析 ONNX 模型**：然后，利用 TensorRT 来解析这个 ONNX 模型并生成针对 NVIDIA GPU 优化的运行时引擎。

#### 步骤一：从 PyTorch 导出到 ONNX

你可以使用 `torch.onnx.export` 方法将你的 PyTorch 模型转换为 ONNX 格式。下面是一个简单的例子：

```python
import torch

# 假设 model 是你已经训练好的 PyTorch 模型
model.eval()  # 设置为评估模式

# 创建一个示例输入张量，确保与模型训练时使用的输入形状一致
dummy_input = torch.randn(1, 3, 192, 192)

# 导出模型
torch.onnx.export(model, dummy_input, "weather_classifier.onnx",
                  input_names=["input"],
                  output_names=["output"],
                  opset_version=11)
```

#### 步骤二：使用 TensorRT 解析 ONNX 并生成 TensorRT 引擎

接下来，你需要使用 TensorRT 的 Python API 或者 C++ API 来加载这个 ONNX 文件，并生成 TensorRT 引擎。这里以 Python API 为例：

首先，确保你已经安装了 TensorRT 和相关的依赖项。可以参考 [NVIDIA TensorRT 官方文档](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) 获取详细的安装指南。

然后，使用以下代码片段将 ONNX 模型转换为 TensorRT 引擎：

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 30  # 1GB
        builder.max_batch_size = 1
        # Parse model file
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Build and return an engine.
        engine = builder.build_cuda_engine(network)
        return engine

engine = build_engine("weather_classifier.onnx")

# 保存引擎
with open("weather_classifier.trt", "wb") as f:
    f.write(engine.serialize())
```

这段代码首先构建了一个 TensorRT 引擎，然后将其序列化并保存为 `.trt` 文件。

### 总结

- **第一步**：使用 `torch.onnx.export()` 将 PyTorch 模型转换为 ONNX 格式。
- **第二步**：使用 TensorRT 的 Python API 加载 ONNX 模型并生成 TensorRT 引擎，最终保存为 `.trt` 文件。

这使得你的模型可以在 NVIDIA GPU 上高效地运行。需要注意的是，为了充分利用 TensorRT 的性能优势，你应该在目标硬件上进行部署和测试。此外，TensorRT 对不同版本的兼容性要求较高，请确保所有相关软件包（如 CUDA、cuDNN、TensorRT 等）版本匹配。

## 封装成接口

当然可以！我们可以将你刚才的 **ONNX 推理脚本封装成一个 Web 接口**，使用 **FastAPI**（推荐）或 **Flask**。FastAPI 更现代、性能更好、支持异步、自带文档，所以我将用 **FastAPI** 为你演示。

---

## 🚀 项目目标

- 接收用户上传的图片（如天气图片）
- 使用 ONNX Runtime 进行推理
- 返回预测结果（如：`sunrise`）

---

## 📦 所需依赖

请先安装以下依赖：

```bash
pip install fastapi uvicorn onnx onnxruntime pillow torchvision
```

---

## 📁 文件结构建议

```
weather_classifier_api/
│
├── app.py                # FastAPI 主程序
├── weather_classifier.onnx   # ONNX 模型文件
└── requirements.txt      # 依赖文件（可选）
```

---

## 📄 `app.py`：FastAPI Web 接口代码

```python
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms
import io

app = FastAPI()

# 类别名称（必须与训练时一致）
class_names = ['cloudy', 'rain', 'shine', 'sunrise']

# 加载 ONNX 模型
ort_session = ort.InferenceSession("weather_classifier.onnx")

# 图像预处理（必须与训练时一致）
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(192),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # 读取图像文件
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # 预处理图像
    image_tensor = transform(image).unsqueeze(0)  # 添加 batch 维度
    image_np = image_tensor.numpy()  # ONNX Runtime 接受 numpy 输入

    # 推理
    outputs = ort_session.run(
        None,
        {'input': image_np}  # 输入名必须与导出 ONNX 时一致
    )

    # 解析结果
    predicted_idx = np.argmax(outputs[0], axis=1)[0]
    predicted_class = class_names[predicted_idx]

    return {"prediction": predicted_class}
```

---

## 🚀 启动 API 服务

在终端中运行以下命令启动服务：

```bash
uvicorn app:app --reload
```

服务启动后，默认地址为：

👉 http://127.0.0.1:8000

Swagger 文档地址：

👉 http://127.0.0.1:8000/docs

---

## 🧪 测试接口（Swagger UI）

1. 打开浏览器访问：http://127.0.0.1:8000/docs
2. 点击 `/predict/` 接口，点击 `Try it out`
3. 选择图片文件上传，点击 `Execute`
4. 将返回预测结果，如：

```json
{
  "prediction": "sunrise"
}
```

---

## 📦 可选：打包为可执行文件（如 `.exe`）

如果你希望把这个 API 打包成一个独立的 `.exe` 或 `.app` 文件，我可以帮你用 **PyInstaller** 打包。只需告诉我你使用的是 **Windows、Mac 还是 Linux**。

---

## ✅ 总结

| 功能     | 实现方式        |
| -------- | --------------- |
| Web 框架 | FastAPI         |
| 图像上传 | `UploadFile`    |
| 推理引擎 | ONNX Runtime    |
| 模型格式 | ONNX            |
| 返回结果 | JSON 格式       |
| 接口文档 | 自带 Swagger UI |

---

如果你还想进一步：

- 把这个接口部署到服务器（如 Nginx + Gunicorn）
- 支持多图批量推理
- 添加身份验证（Token）
- 加入日志记录和性能监控
- 用 Docker 容器化部署

我也可以继续帮你完成 😎

是否需要我帮你生成一个 **可执行文件版本** 或 **Docker 部署脚本**？