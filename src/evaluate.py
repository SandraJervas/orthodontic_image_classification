import torch
from torchvision import models, datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load("models/resnet18.pth"))
model.eval()

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
test_data = datasets.ImageFolder("data/processed/split/test", transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

report = classification_report(y_true, y_pred, target_names=test_data.classes)
with open("outputs/classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=test_data.classes, yticklabels=test_data.classes)
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
