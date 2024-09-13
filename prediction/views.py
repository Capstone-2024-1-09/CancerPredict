import torch
import os
from django.shortcuts import render
from django.conf import settings
from .models_files.saint_model import SAINT
import numpy as np

# 모델 초기화 및 로드 (서버 시작 시 한 번만 로드)
MODEL_PATH = os.path.join(settings.BASE_DIR, 'prediction', 'models_files', 'saint_model.pth')

# 모델 파라미터 (Colab에서 설정한 것과 동일하게)
INPUT_DIM = 7  # Age, Gender, BMI, Smoking, PhysicalActivity, AlcoholIntake, CancerHistory (7개 특성)
HIDDEN_DIM = 128
OUTPUT_DIM = 2  # 이진 분류 (암 진단 여부)

model = SAINT(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(state_dict)
model.eval()


def preprocess_input(age, gender, bmi, smoking, physical_activity, alcohol_intake, cancer_history):
    
    # 사용자 입력을 모델이 요구하는 형태로 전처리하는 함수
    
    # 데이터 전처리
    age = float(age)
    gender = float(gender)  # 성별은 0과 1로 인코딩 (0: 남성, 1: 여성)
    bmi = float(bmi)
    smoking = float(smoking)  # 흡연 여부는 0과 1로 인코딩
    physical_activity = float(physical_activity)
    alcohol_intake = float(alcohol_intake)
    cancer_history = float(cancer_history)  # 암 병력 여부는 0과 1로 인코딩

    # 특성 배열 생성
    features = np.array([age, gender, bmi, smoking, physical_activity, alcohol_intake, cancer_history])
    return torch.tensor(features, dtype=torch.float32)


def predict_cancer(request):
    if request.method == 'POST':
        # 사용자 입력을 POST 요청에서 가져옵니다.
        age = request.POST.get('Age')
        gender = request.POST.get('Gender')
        bmi = request.POST.get('BMI')
        smoking = request.POST.get('Smoking')
        physical_activity = request.POST.get('PhysicalActivity')
        alcohol_intake = request.POST.get('AlcoholIntake')
        cancer_history = request.POST.get('CancerHistory')

        # 입력값 유효성 검사
        if not age or not gender or not bmi or not smoking or not physical_activity or not alcohol_intake or not cancer_history:
            return render(request, 'prediction/predict.html', {'error': '모든 필드를 입력해주세요.'})

        # 입력값 전처리
        try:
            input_tensor = preprocess_input(age, gender, bmi, smoking, physical_activity, alcohol_intake,
                                            cancer_history)
        except ValueError:
            return render(request, 'prediction/predict.html', {'error': '잘못된 입력 형식입니다.'})

        # 배치 차원 추가 (1개의 샘플을 예측하므로)
        input_tensor = input_tensor.unsqueeze(0)

        # 모델 예측 수행
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            cancer_probability = probabilities[0][1].item()  # 클래스 1(암이 있을 확률)

        # 암이 있을 확률만 출력
        result = f"암 발병 확률: {cancer_probability * 100:.2f}%"

        return render(request, 'prediction/result.html', {'result': result})

    return render(request, 'prediction/predict.html')