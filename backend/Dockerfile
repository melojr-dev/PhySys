FROM python:3.11-slim
ARG APP_VERSION=dev
ARG GIT_SHA=unknown
ARG BUILD_TIME=unknown
LABEL org.opencontainers.image.title="PhysioAI Backend" \
      org.opencontainers.image.version="$APP_VERSION" \
      org.opencontainers.image.revision="$GIT_SHA" \
      org.opencontainers.image.created="$BUILD_TIME"
ENV PHYSIOAI_VERSION=$APP_VERSION \
    PHYSIOAI_GIT_SHA=$GIT_SHA \
    PHYSIOAI_BUILD_TIME=$BUILD_TIME \
    PHYSIOAI_DATA_DIR=/app/data \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 physioai \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin physioai \
    && mkdir -p /app/data \
    && chown -R physioai:physioai /app/data /home/physioai
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
# Pr�-carrega o modelo Heavy durante o build enquanto o filesystem do Python
# ainda � grav�vel. Em runtime o backend roda como usu�rio n�o-root.
RUN python -c "import mediapipe as mp; p=mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2); p.close(); print('MediaPipe Pose Heavy pronto')"
COPY --chown=physioai:physioai . .
USER physioai
EXPOSE 8000
CMD ["python", "server.py"]
