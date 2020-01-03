import torch
import json
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import Image

with open("imagenet_class_index.json") as f:
    class_idx = json.load(f)
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

# - 定义模型，加载权重
model = mobilenet_v2()
# print(model)
if torch.cuda.is_available():
    model.load_state_dict(torch.load("mobilenet_v2-b0353104.pth"))
else:
    model.load_state_dict(torch.load("mobilenet_v2-b0353104.pth", map_location="cpu"))

#
model.eval()

# - 处理输入数据

process_img = transforms.Compose([
    transforms.Scale(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img = Image.open('ILSVRC2012_val_00000006.JPEG').convert('RGB')
input_data = process_img(img).unsqueeze(0)

# - 前向得到预测结果
out = model(input_data)
print(torch.argmax(out, dim=1))
print(out[0].sort()[1][-5:])

#
for idx in out[0].sort()[1][-5:]:
    print(idx2label[idx])
