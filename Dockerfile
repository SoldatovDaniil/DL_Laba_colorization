FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скачать веса моделей на этапе сборки для офлайн-запуска
RUN python -c "from torchvision.models import vgg16; vgg16()"
COPY . .
RUN python -c "from colorizers import eccv16, siggraph17; eccv16(pretrained=True); siggraph17(pretrained=True)"

CMD ["python", "RunCheckerTest.py"]
