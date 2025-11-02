# app.py - Emotion Detection Web App (uses your trained model)

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import os
import sqlite3

# -------------------------------------------------
# Flask setup
# -------------------------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------------------------
# 1. MODEL DEFINITION (must be EXACTLY the same as in model.py)
# -------------------------------------------------


class SimpleEmotionModel(nn.Module):
    def __init__(self):
        super(SimpleEmotionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 7)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(self.relu(self.batch_norm3(self.conv3(x))))
        x = x.view(-1, 128 * 6 * 6)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# -------------------------------------------------
# 2. LOAD THE TRAINED MODEL
# -------------------------------------------------
device = torch.device('cpu')
model = SimpleEmotionModel().to(device)

# NOTE: if you ever switch to the pre-trained 85% model, comment the line above
# and use the block from the previous answer.

model_path = 'emotion_model.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file '{model_path}' not found. Run `python model.py` first.")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -------------------------------------------------
# 3. PRE-PROCESSING (same as training)
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# -------------------------------------------------
# 4. DATABASE SETUP
# -------------------------------------------------
conn = sqlite3.connect('users.db')
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT,
        image_path TEXT,
        emotion TEXT
    )
''')
conn.commit()
conn.close()

# -------------------------------------------------
# 5. PREDICTION HELPERS
# -------------------------------------------------


def predict_image(img: Image.Image) -> str:
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        idx = output.argmax().item()
    return emotions[idx]

# -------------------------------------------------
# 6. ROUTES
# -------------------------------------------------


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Please enter your name'})

    image_path = None
    img = None

    # ---- Uploaded file -------------------------------------------------
    if 'file' in request.files and request.files['file'].filename:
        file = request.files['file']
        img = Image.open(file.stream).convert('RGB')
        image_path = os.path.join(UPLOAD_FOLDER, f"{name}_uploaded.jpg")
        img.save(image_path, 'JPEG')
        print(f"[DEBUG] Saved uploaded image: {image_path}")

    # ---- Camera capture (base64) ---------------------------------------
    elif 'image' in request.form:
        b64 = request.form['image'].split(',', 1)[1]
        # fix padding
        b64 += '=' * ((4 - len(b64) % 4) % 4)
        try:
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            image_path = os.path.join(UPLOAD_FOLDER, f"{name}_camera.jpg")
            img.save(image_path, 'JPEG')
            print(f"[DEBUG] Saved camera image: {image_path}")
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {e}'})

    else:
        return jsonify({'error': 'No image provided'})

    # ---- Predict -------------------------------------------------------
    emotion = predict_image(img)

    # ---- Save to DB ----------------------------------------------------
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO users (name, image_path, emotion) VALUES (?, ?, ?)",
        (name, image_path, emotion)
    )
    conn.commit()
    conn.close()

    return jsonify({'emotion': emotion})


# -------------------------------------------------
# 7. RUN
# -------------------------------------------------
if __name__ == '__main__':
    print("Go to http://127.0.0.1:5000 in your browser")
    app.run(debug=True)
