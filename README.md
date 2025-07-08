
# 👮‍♂️ Criminal Face Detection using OpenCV

This project is a real-time criminal face recognition system built with Python and OpenCV. It uses the webcam to detect and recognize faces based on a trained image dataset using the LBPH (Local Binary Pattern Histogram) algorithm and Haar Cascade Classifier.

---

## 📌 Features

- Real-time face detection from webcam
- Recognition of trained faces using LBPH algorithm
- Logging of detected faces with timestamp
- Customizable face database
- Works offline without internet


## 🧠 How It Works

1. The program uses a private set of face images (not included in this repo) for training.
2. Face detection is done using Haar Cascades.
3. Face recognition is performed using the LBPH algorithm from OpenCV.
4. Detected and recognized names are shown on screen and logged in a text file.

⚠️ **Note:** The face dataset used for training is private and not uploaded for privacy reasons.

## ▶️ How to Run

1. Make sure you have Python installed.
2. Install required libraries:

```bash
pip install opencv-python opencv-contrib-python numpy
````

3. Run the Python file:

```bash
python "criminal face.py"
```

4. Press `Q` to quit the webcam window.

---

## 📂 Project Structure

```
criminal-face-detection/
├── criminal face.py               # Main code file
├── criminal/                      # (Private) Folder containing training images
├── criminal_detection_log.txt     # Log file for detections
└── README.md                      # Project description
```

> Note: The `criminal/` folder is where face images are stored. This folder is **not included** in the repo.
## 📌 Requirements

* Python 3.x
* OpenCV
* NumPy
