import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import torch
import torch.nn as nn
from torchvision import transforms, models


# Settings
model_path = "model_4class.pth"
class_names = ["both", "cat", "dog", "neither"]
num_classes = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Transform

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# prediction function
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)

    return class_names[pred.item()]

# GUI

root = tk.Tk()
root.title("Image Classifier (Cat/Dog/Both/Neither)")
root.geometry("500x550")

label_title = tk.Label(root, text="4-Class Image Classifier", font=("Arial", 18, "bold"))
label_title.pack(pady=10)

img_label = tk.Label(root)
img_label.pack(pady=10)

result_label = tk.Label(root, text="Prediction: ---", font=("Arial", 16))
result_label.pack(pady=10)

def open_file():
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if file_path:
        # Show image
        img = Image.open(file_path).convert("RGB")
        img_resized = img.resize((350, 350))
        img_tk = ImageTk.PhotoImage(img_resized)
        img_label.config(image=img_tk)
        img_label.image = img_tk

        # Predict
        pred = predict_image(file_path)
        result_label.config(text=f"Prediction: {pred}")

btn = tk.Button(root, text="Select Image", command=open_file, font=("Arial", 14))
btn.pack(pady=20)

root.mainloop()