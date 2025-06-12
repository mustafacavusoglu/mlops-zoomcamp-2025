FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Pipenv yükle
RUN pip install --upgrade pip && pip install pipenv

# Pipfile ve Pipfile.lock dosyalarını kopyala
COPY ["Pipfile", "Pipfile.lock", "./"]

# pipenv ile sistem ortamına kurulum yap
RUN pipenv install --deploy --system

# Uygulama dosyasını kopyala
COPY scoring.py .

# Uygulamayı çalıştır
ENTRYPOINT ["python", "scoring.py", "2023", "3" ]
