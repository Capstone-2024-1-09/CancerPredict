# admin.py

from django.contrib import admin
from .models import YourModel  # models.py에서 정의한 모델 임포트

# 모델을 관리자(admin)에 등록
admin.site.register(YourModel)
