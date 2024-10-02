# Python 3.9
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요 패키지 설치
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . /app/

# 정적 파일 수집 (필요시)
RUN python manage.py collectstatic --noinput

# Gunicorn을 사용해 Django 애플리케이션 실행
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "Capstone.wsgi:application"]
