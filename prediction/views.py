# prediction/views.py

import torch
import os
from django.shortcuts import render
from django.conf import settings
from .models_files.saint_model import SAINT
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# 모델 초기화 및 로드 (서버 시작 시 한 번만 로드)
MODEL_PATH = os.path.join(settings.BASE_DIR, 'prediction', 'models_files', 'saint_model.pth')

# 모델 파라미터
INPUT_DIM = 7
HIDDEN_DIM = 64
OUTPUT_DIM = 2

model = SAINT(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# 데이터 로드 및 전처리
EXPLAINER_PATH = os.path.join(settings.BASE_DIR, 'prediction', 'data', 'The_Cancer_data_1500_V3.csv')
data = pd.read_csv(EXPLAINER_PATH)
Y = data['Diagnosis'].values
X = data[['Age', 'Gender', 'BMI', 'Smoking', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)

# Explainer 선언 및 SHAP 값 계산
explainer = shap.GradientExplainer(model, X_train_tensor)
shap_values_train = explainer.shap_values(X_train_tensor)
shap_values_train_2d = shap_values_train[:, :, 1]
shap_mean_abs_values = np.abs(shap_values_train_2d).mean(axis=0)

# Feature 목록
columns = ['Age', 'Gender', 'BMI', 'Smoking', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']
rank_name = ['나이', '성별', 'BMI', '흡연', '운동량', '음주', '암 발병 경험 유무']

# SHAP 평균값 DataFrame 생성 및 정렬
shap_mean_abs_df = pd.DataFrame({
    'Feature': columns,
    'Mean_SHAP_Value': shap_mean_abs_values,
    'Rank': rank_name
})
shap_mean_abs_df = shap_mean_abs_df[shap_mean_abs_df['Feature'] != 'Gender']
shap_mean_abs_df_sorted = shap_mean_abs_df.sort_values(by='Mean_SHAP_Value', ascending=False)

def preprocess_input(age, gender, bmi, smoking, physical_activity, alcohol_intake, cancer_history):
    try:
        features = [
            float(age),
            float(gender),
            float(bmi),
            float(smoking),
            float(physical_activity),
            float(alcohol_intake),
            float(cancer_history)
        ]
    except ValueError:
        raise ValueError("잘못된 입력 형식입니다.")
    return torch.tensor(features, dtype=torch.float32)

def predict_cancer(request):
    if request.method == 'POST':
        # 사용자 입력값 가져오기
        age = request.POST.get('Age')
        gender = request.POST.get('Gender')
        bmi = request.POST.get('BMI')
        smoking = request.POST.get('Smoking')
        physical_activity = request.POST.get('PhysicalActivity')
        alcohol_intake = request.POST.get('AlcoholIntake')
        cancer_history = request.POST.get('CancerHistory')

        # 입력값 유효성 검사
        if not all([age, gender, bmi, smoking, physical_activity, alcohol_intake, cancer_history]):
            return render(request, 'predict.html', {'error': '모든 필드를 입력해주세요.'})

        # 입력값 전처리
        try:
            input_tensor = preprocess_input(age, gender, bmi, smoking, physical_activity, alcohol_intake, cancer_history)
        except ValueError as e:
            return render(request, 'predict.html', {'error': str(e)})

        # 배치 차원 추가
        input_tensor = input_tensor.unsqueeze(0)

        # 모델 예측
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            cancer_probability = probabilities[0][1].item()

        result = f"암 발병 확률: {cancer_probability * 100:.2f}%"

        # 초기 컨텍스트 준비
        context = {
            'result': result,
            'rank_features': rank_name
        }

        # 암 발병 확률이 20% 이상일 때만 SHAP 값을 계산하고, 그래프를 포함
        if cancer_probability * 100 >= 20:
            # SHAP 값 계산
            shap_values_input = explainer.shap_values(input_tensor)
            shap_values_input_2d = shap_values_input[:, :, 1]
            shap_mean_abs_input_values = np.abs(shap_values_input_2d).mean(axis=0)


            shap_information = [
                '나이 - 병원에서 정기적으로 검진 받기 , 40대 이상: 위암, 대장암, 간암, 폐암 등의 검진을 주기적으로 받아야 합니다., 50대 이상: 특히 대장암, 전립선암, 폐암 검진의 필요성이 높습니다. 고위험군은 전문가와 상의하여 개인화된 검진 일정을 따르는 것이 좋습니다.',
                '성별 - 성별에 맞는 정기 검진과 생활 습관 관리 중요합니다. , 남성 : 전립선암, 폐암, 간암 등이 높은 발생률을 보입니다. 특히 폐암과 간암 예방을 위해 금연과  금주가  필수적입니다. , 여성 : 유방암, 자궁경부암, 난소암 등이 주요 암으로 꼽히며, 정기적인 유방암 및 자궁경부암 검진이 필요합니다.',
                'BMI - 비만은 여러 암의 주요 위험 요인이므로 적정 체중 유지가 중요합니다. BMI가 높은 경우 체중 감량을 위해 식이 조절과 함께 규칙적인 운동을 병행하는 것이 암 예방에 효과적입니다.  고섬유질 식품, 채소, 과일 위주의 식단을 유지하고, 고열량 음식 섭취를 제한합니다.',
                '흡연 - 흡연은 암의 주요 원인 중 하나로, 금연이 가장 효과적인 암 예방 방법입니다.  폐암, 구강암, 후두암, 방광암 등의 위험을 줄이기 위해 금연 프로그램을 통해 금연을 실천해야 합니다. 니코틴 대체 요법이나 상담 치료를 통해 금연을 성공적으로 유지하는 것이 중요합니다.',
                '운동량 - 규칙적인 신체 활동은 암 예방에 큰 도움이 됩니다. 주기적인 운동은 대장암, 유방암, 자궁내막암 등의 발생 위험을 낮춥니다.  매일 최소 30분 이상의 중간 강도의 운동(걷기, 자전거 타기 등)을 실천하고, 주 2회 이상 근력 운동을 병행합니다. 운동은 체중 조절과 면역력 강화에 도움을 주며, 스트레스 해소에도 긍정적인 영향을 미칩니다.',
                '음주 - 음주는 다양한 암의 위험을 높이므로 음주량을 줄이거나 금주하는 것이 암 예방에 매우 중요합니다. 간암, 구강암, 식도암, 유방암 등 여러 암이 음주와 연관이 있으므로, 가능한 음주 빈도를 줄이거나 주당 권장 음주량을 준수하는 것이 좋습니다.',
                '암 발병 이력 - 정기적인 추적 관찰과 생활 습관 개선이 중요합니다. 암 재발 방지를 위해 병원에서 권장하는 정기 검진을 반드시 따르고, 체력 증진을 위한 규칙적인 운동과 균형 잡힌 식단이 필요합니다. 면역력을 강화하는 음식 섭취와 스트레스 관리도 재발 예방에 도움이 됩니다.'
            ]

            # SHAP 값 DataFrame 생성
            shap_mean_abs_input_df = pd.DataFrame({
                'Feature': columns,
                'Mean_SHAP_Value': shap_mean_abs_input_values,
                'information': shap_information
            })

            shap_mean_abs_input_df = shap_mean_abs_input_df[shap_mean_abs_input_df['Feature'] != 'Gender']
            shap_mean_abs_input_df_sorted = shap_mean_abs_input_df.sort_values(by='Mean_SHAP_Value', ascending=False)

            # 사용자 입력값에 따른 SHAP값 그래프
            plt.figure(figsize=(8, 6))
            plt.barh(shap_mean_abs_input_df_sorted['Feature'], shap_mean_abs_input_df_sorted['Mean_SHAP_Value'],
                     color='lightblue')
            plt.xlabel('SHAP Value (Mean Absolute)')
            plt.title('SHAP value')
            plt.tight_layout()

            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            shap_plot = base64.b64encode(image_png).decode('utf-8')
            plt.close()

            # 원인과 해결법
            main_causes = shap_mean_abs_input_df_sorted.iloc[0]['Feature']
            solution = shap_mean_abs_input_df_sorted.iloc[0]['information']

            # 컨텍스트에 SHAP 관련 데이터 추가
            context.update({
                'cause': main_causes,
                'solution': solution,
                'shap_plot': shap_plot,
            })
        else:
            # 암 발병 확률이 20% 미만일 때 기본 메시지 설정
            main_causes = "없습니다."
            solution = "주기적인 암 검진은 암 예방에 매우 효과적입니다."

            context.update({
                'cause': main_causes,
                'solution': solution,
            })

        return render(request, 'result.html', context)

    # GET 요청 처리
    return render(request, 'predict.html')

def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

#def result(request):
#    return render(request, 'result.html')