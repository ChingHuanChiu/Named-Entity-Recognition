FROM ubuntu:20.04
FROM python:3.7-slim


ENV TZ=Asia/Taipei

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r ./requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]