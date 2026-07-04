import json
import os
import joblib
import boto3
import pandas as pd

# إعدادات AWS S3 والملف (تم ضبطها بناءً على الاسم في صورتكِ)
BUCKET_NAME = "heartrate-model-storage" 
MODEL_KEY = "heart_rate_model.pkl"
LOCAL_MODEL_PATH = "/tmp/heart_rate_model.pkl"

model = None

def load_model_from_s3():
    """تحميل الموديل من S3 إلى مسار مؤقت في Lambda مرة واحدة فقط عند الـ Warm Start"""
    global model
    if model is None:
        if not os.path.exists(LOCAL_MODEL_PATH):
            print(f"Downloading model {MODEL_KEY} from S3 bucket {BUCKET_NAME}...")
            s3 = boto3.client('s3')
            s3.download_file(BUCKET_NAME, MODEL_KEY, LOCAL_MODEL_PATH)
            print("Download completed.")
        
        print("Loading model into memory...")
        model = joblib.load(LOCAL_MODEL_PATH)
        print("Model loaded successfully.")

def handler(event, context):
    try:
        # استدعاء دالة التحميل من S3
        load_model_from_s3()
        
        # 1. استقبال البيانات القادمة من الطلب
        body = event.get("body", "{}")
        if isinstance(body, str):
            data = json.loads(body)
        else:
            data = body
            
        if "features" not in data:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'features' key in the input JSON request."})
            }
            
        input_features = data["features"]
        
        # 2. تحويل البيانات وعمل التوقع
        X_input = pd.DataFrame(input_features)
        predictions = model.predict(X_input)
        
        # 3. إرسال الاستجابة
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "predictions": predictions.tolist()
            })
        }
        
    except Exception as e:
        print(f"Exception caught: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Prediction failed: {str(e)}"})
        }