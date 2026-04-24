from flask import Flask, render_template, request
import pickle
import cv2
import numpy as np
import os
from skimage.feature import hog
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================
# LOAD ALL MODELS
# =========================
svm_model = pickle.load(open("svm_hog_model.pkl", "rb"))
rf_model  = pickle.load(open("rf_hog_model.pkl", "rb"))
lr_model  = pickle.load(open("lr_hog_model.pkl", "rb"))
hog_scaler = pickle.load(open("hog_scaler.pkl", "rb"))

# =========================
# DEFINE PYTORCH CNN MODEL
# =========================
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


cnn_model = CNNModel()
cnn_model.load_state_dict(torch.load("cnn_model.pth", map_location=torch.device('cpu')))
cnn_model.eval()


# =========================
# HOG FEATURE FUNCTION
# =========================
def extract_hog_features(img):
    img = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return features.reshape(1, -1)


# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        image_file = request.files["image"]
        model_choice = request.form["model"]

        # Save uploaded image
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
        image_file.save(image_path)

        # Read image
        img = cv2.imread(image_path)

        # ---------------- CNN ----------------
        if model_choice == "cnn":

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])

            img_tensor = transform(img)
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                output = cnn_model(img_tensor)
                result = output.item()

            prediction = "Dog 🐶" if result > 0.5 else "Cat 🐱"

        # ---------------- ML MODELS ----------------
        else:
            hog_features = extract_hog_features(img)

            # Scale ONLY for SVM and Logistic Regression
            if model_choice in ["svm", "lr"]:
                hog_features = hog_scaler.transform(hog_features)

            if model_choice == "svm":
                result = svm_model.predict(hog_features)[0]
            elif model_choice == "rf":
                result = rf_model.predict(hog_features)[0]
            elif model_choice == "lr":
                result = lr_model.predict(hog_features)[0]

            prediction = "Dog 🐶" if result == 1 else "Cat 🐱"

    return render_template("index.html", prediction=prediction)


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True,port=5000)
