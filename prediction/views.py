import torch
import os
from django.shortcuts import render
from django.conf import settings
from .models_files.saint_model import SAINT
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
from functools import lru_cache
import logging

# 로그 설정
logger = logging.getLogger(__name__)

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

EXPLAINER_PATH = os.path.join(settings.BASE_DIR,'prediction','data','The_Cancer_data_1500_V3.csv')
data = pd.read_csv(EXPLAINER_PATH)
Y = data['Diagnosis'].values
X = data[['Age', 'Gender', 'BMI', 'Smoking', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# train 데이터 torch 형태로 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)

# SHAP Explainer
@lru_cache(maxsize=1)
def get_explainer():
    try:
        return shap.GradientExplainer(SAINT(INPUT_DIM, HIDDEN_DIM, 1), X_train_tensor)
    except Exception as e:
        logger.error(f"SHAP Explainer 초기화 실패: {e}")
        return None

#for i in range(0,shap_mean_abs_df_sorted.shape[0]):
    #print(shap_mean_abs_df_sorted.iloc[i,0])


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
            return render(request, 'predict.html', {'error': '모든 필드를 입력해주세요.'})

        # 입력값 전처리
        try:
            input_tensor = preprocess_input(age, gender, bmi, smoking, physical_activity, alcohol_intake,
                                            cancer_history)
        except ValueError:
            return render(request, 'predict.html', {'error': '잘못된 입력 형식입니다.'})

        # 배치 차원 추가 (1개의 샘플을 예측하므로)
        input_tensor = input_tensor.unsqueeze(0)

        # 모델 예측 수행
        try:
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                cancer_probability = probabilities[0][1].item()  # 클래스 1(암이 있을 확률)
        except Exception as e:
            logger.error(f"모델 예측 실패: {e}")
            return render(request, 'predict.html', {'error': '모델 예측 중 오류가 발생했습니다.'})

        # 입력값을 explanier에 학습
        explainer = get_explainer()
        if explainer is None:
            return render(request, 'predict.html', {'error': 'SHAP Explainer 초기화에 실패했습니다.'})

        try:
            shap_values_input = explainer.shap_values(input_tensor)
            shap_values_input_2d = shap_values_input.squeeze(2)  # 차원 축소
            shap_mean_abs_input_values = np.abs(shap_values_input_2d).mean(axis=0)
        except Exception as e:
            logger.error(f"SHAP 계산 실패: {e}")
            return render(request, 'predict.html', {'error': 'SHAP 계산 중 오류가 발생했습니다.'})

        columns = ['Age', 'Gender', 'BMI', 'Smoking', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']
        shap_mean_abs_input_df = pd.DataFrame({
            'Feature': columns,
            'Mean_SHAP_Value': shap_mean_abs_input_values
        })
        shap_mean_abs_input_df_sorted = shap_mean_abs_input_df.sort_values(by='Mean_SHAP_Value', ascending=False)
        shap_mean_abs_input_df_final = shap_mean_abs_input_df_sorted.set_index('Feature')

        # 기존 SHAP 평균 값 로드
        try:
            shap_values_train = explainer.shap_values(X_train_tensor)
            shap_values_train_2d = np.squeeze(shap_values_train)
            shap_mean_abs_values = np.abs(shap_values_train_2d).mean(axis=0)

            shap_mean_abs_df = pd.DataFrame({
                'Feature': columns,
                'Mean_SHAP_Value': shap_mean_abs_values
            })
            shap_mean_abs_df_sorted = shap_mean_abs_df.sort_values(by='Mean_SHAP_Value', ascending=False)
            shap_mean_abs_df_final = shap_mean_abs_df_sorted.set_index('Feature')
        except Exception as e:
            logger.error(f"기존 SHAP 평균 값 로드 실패: {e}")
            return render(request, 'predict.html', {'error': 'SHAP 데이터 로드 중 오류가 발생했습니다.'})


        # 암이 있을 확률만 출력
        predictResult = f"암 발병 확률: {cancer_probability * 100:.2f}%"

        shap_diff = []
        shap_diff_information = ['Age Diff', 'Gender Diff', 'BMI Diff', 'Smoking Diff', 'PhysicalActivity Diff', 'AlcoholIntake Diff', 'CancerHistory Diff']
        
        
        

        print(shap_mean_abs_df_sorted)
        print(shap_mean_abs_input_df_sorted)


        for i in range(0,len(columns)):
            shap_diff.append(shap_mean_abs_input_df_final.loc[columns[i],'Mean_SHAP_Value']-shap_mean_abs_df_final.loc[columns[0],'Mean_SHAP_Value'])
        
        shap_diff_df = pd.DataFrame({
            'Feature': columns,
            'Shap_Diff': shap_diff,
            'Shap_Diff_Information': shap_diff_information
        })

        shap_diff_df_sorted = shap_diff_df.sort_values(by='Shap_Diff', ascending=False)

        print(shap_diff_df_sorted)




        #for i in range(0,shap_mean_abs_df_sorted.shape[0]):
            #input_features.append(shap_mean_abs_df_sorted.iloc[i,0])

        #print(input_features)
        #print(input_shap_diff)
        
            
        
        return render(request, 'result.html', {'result': predictResult})

    return render(request, 'predict.html')
def index(request):
    return render(request, 'index.html')
def result(request):
    return render(request, 'result.html')
def about(request):
    return render(request, 'about.html')