# 베이스 이미지로 Python 3.11 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 종속성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 복사
COPY . .

# Django collectstatic 실행
RUN python manage.py collectstatic --noinput

# 포트 설정
EXPOSE 8000

# Gunicorn으로 애플리케이션 실행
CMD ["gunicorn", "Capstone.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3"]
