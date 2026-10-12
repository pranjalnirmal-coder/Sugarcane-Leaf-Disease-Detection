# Sugarcane Leaf Disease Detection using AI-Based Image Processing
OOSE project - K. J. Somaiya School of Engineering (Pranjal Nirmal, 16014225812)

## Run
    pip install -r requirements.txt
    streamlit run app.py
Demo logins: admin/admin123, expert/expert123 (or register as farmer).

## Train a real model
Put images in `dataset/<ClassName>/` (Healthy, Red Rot, Smut, Brown Rust, Leaf Scald, Red Stripe, Mosaic, Eye Spot), then:
    python train.py --data dataset --epochs 10
This writes `models/sugarcane_cnn.keras`, `models/classes.json`, `models/evaluation.json` (precision/recall/F1/confusion matrix). The app switches from DEMO mode to the CNN automatically.

## Report mapping
| Report chapter | Code |
|---|---|
| Ch.1.5 GUI use case | app.py (Analyze Leaf page) |
| Ch.2 SRS features 1-6 | login (db.py), upload/validate (preprocess.py), detection (model.py), Grad-CAM (model.py), advisory (advisory.py), history (db.py) |
| Ch.2.5 database tables | core/db.py |
| Ch.4 classes | User, LeafImage, ImagePreprocessor, CNNModel, GradCAMResult, Advisory, PredictionHistory map to db tables + core modules |
| Ch.5 layered design | UI (app.py) / application / AI (core/model.py, preprocess.py) / data (core/db.py) |
