from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import os
import sqlite3

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === LIGHTWEIGHT MODEL ===
class TinyEmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 12 * 12, 64)
        self.fc2 = nn.Linear(64, 7)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 12 * 12)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# === LOAD MODEL ONCE ===
print("Loading model...")
device = torch.device("cpu")
model = TinyEmotionCNN().to(device)

# Load your trained weights (emotion_model.pth)
try:
    model.load_state_dict(torch.load('emotion_model.pth', map_location=device))
    print("Model loaded!")
except:
    print("No model found — using random weights")

model.eval()

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# === DB ===
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users 
             (name TEXT, image_path TEXT, emotion TEXT)''')
conn.commit()

def predict_image(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        idx = output.argmax().item()
    return emotions[idx]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name'].strip()
    if not name:
        return jsonify({'error': 'Enter name'})

    image_path = None

    # === UPLOAD ===
    if 'file' in request.files and request.files['file'].filename:
        file = request.files['file']
        img = Image.open(file.stream).convert('RGB')
        image_path = os.path.join(UPLOAD_FOLDER, f"{name}_up.jpg")
        img.save(image_path, 'JPEG')

    # === CAMERA ===
    elif 'image' in request.form:
        data = request.form['image'].split(',')[1]
        data += '=' * (-len(data) % 4)
        img_bytes = base64.b64decode(data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        image_path = os.path.join(UPLOAD_FOLDER, f"{name}_cam.jpg")
        img.save(image_path, 'JPEG')

    else:
        return jsonify({'error': 'No image'})

    # === PREDICT ===
    emotion = predict_image(img)

    # === SAVE TO DB ===
    c.execute("INSERT INTO users VALUES (?, ?, ?)", (name, image_path, emotion))
    conn.commit()

    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run()
