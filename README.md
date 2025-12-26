# 🎭 Age & Gender Detection

Real-time age and gender detection using OpenCV's Deep Neural Networks (DNN) module and pre-trained Caffe models.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Description

This project detects faces in real-time from webcam feed and estimates:
- **Age range** (8 categories: 0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60-100)
- **Gender** (Male/Female)

Built as a personal learning project to explore computer vision and deep learning deployment with OpenCV.

## ✨ Features

- ✅ Real-time face detection
- ✅ Age estimation (8 age ranges)
- ✅ Gender classification
- ✅ Clean OOP architecture
- ✅ Color-coded bounding boxes (blue for male, pink for female)
- ✅ Confidence scores display

## 🛠️ Technologies

- **Python 3.8+**
- **OpenCV** (cv2) - Computer Vision and DNN module
- **NumPy** - Array operations
- **Pre-trained Caffe models**:
  - Face Detection: SSD (Single Shot Detector)
  - Age Estimation: CNN trained on Adience dataset
  - Gender Classification: CNN trained on Adience dataset

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/Fanfulla/cv_webcam.git
cd cv_webcam
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 3. Install dependencies
```bash
pip install opencv-contrib-python numpy
```

### 4. Download pre-trained models

The models are already included in the `models/` directory:
- Face Detection: `deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`
- Age Estimation: `age_deploy.prototxt`, `age_net.caffemodel`
- Gender Classification: `gender_deploy.prototxt`, `gender_net.caffemodel`

## 🚀 Usage
```bash
python3 main.py
```

- The webcam window will open
- Face detection and age/gender estimation run in real-time
- Press **'q'** or **ESC** to exit

## 📁 Project Structure
```
cv_webcam/
├── main.py                 # Entry point
├── face_detector.py        # Face detection class
├── age_estimator.py        # Age estimation class
├── gender_estimator.py     # Gender classification class
├── video_processor.py      # Webcam and video processing
├── models/                 # Pre-trained models
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── gender_deploy.prototxt
│   └── gender_net.caffemodel
├── requirements.txt        # Python dependencies
└── README.md
```

## 🧠 How It Works

1. **Face Detection**: Uses OpenCV's DNN module with a pre-trained SSD model to detect faces
2. **Face Extraction**: Crops detected faces from the frame
3. **Preprocessing**: Resizes faces to 227x227 and applies mean subtraction
4. **Age/Gender Prediction**: Passes preprocessed faces through CNNs
5. **Visualization**: Draws bounding boxes and labels on the original frame

## 🎓 Learning Outcomes

This project helped me learn:
- Object-oriented programming in Python
- OpenCV DNN module and pre-trained model deployment
- Real-time video processing
- Image preprocessing (blob creation, mean subtraction)
- NumPy array operations and image slicing
- Git version control workflow

## 📊 Model Information

### Age Ranges
The model classifies faces into 8 age ranges:
- (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)

### Accuracy
These pre-trained models provide reasonable accuracy for general use but may vary based on:
- Lighting conditions
- Face angle
- Image quality
- Ethnicity representation in training data

## ⚠️ Limitations

- Age estimation returns **ranges**, not exact ages
- Performance depends on lighting and camera quality
- Models may have bias based on training data
- Not suitable for production/critical applications

## 🔮 Future Improvements

- [ ] Add emotion detection
- [ ] Save screenshots on keypress
- [ ] Support for image/video file input
- [ ] Performance metrics and FPS display
- [ ] Model fine-tuning on custom datasets
- [ ] Docker deployment

## 📚 Resources & Credits

- **OpenCV DNN Module**: [OpenCV Documentation](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)
- **Age/Gender Models**: Based on research by Gil Levi and Tal Hassner
- **Face Detection Model**: OpenCV's pre-trained SSD model
- **Dataset**: [Adience Benchmark](https://talhassner.github.io/home/projects/Adience/Adience-data.html)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Salvatore (Fanfulla)**
- GitHub: [@Fanfulla](https://github.com/Fanfulla)
- Project developed as part of AI Engineering Master's program with [Profession.AI](https://www.profession.ai/)

## 🙏 Acknowledgments

- Thanks to the OpenCV community for excellent documentation
- Gil Levi and Tal Hassner for the age/gender models
- Profession.AI for the learning opportunity

---

**Note**: This is a personal learning project. Models and predictions should not be used for critical decision-making.
