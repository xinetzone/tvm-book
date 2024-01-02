from torchvision.io import read_image
from torchvision.models import resnet18, ResNet18_Weights

img = read_image("../assets/grace_hopper_517x606.jpg")

# 步骤1：使用最佳可用权重初始化模型
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()

# 步骤2：初始化推理变换
preprocess = weights.transforms(antialias=True)

# 步骤3：应用推理预处理变换
batch = preprocess(img).unsqueeze(0)

# 步骤4：使用模型并打印预测类别
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")
