# The Playwright image ships Chromium and its system libraries already. Render's
# native Python runtime cannot apt-get them, and PDF export needs a real
# browser, so Docker is the practical choice here rather than a preference.
FROM mcr.microsoft.com/playwright/python:v1.47.0-jammy

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Chromium is in the base image; this just makes sure the version Playwright
# expects is the one present.
RUN python -m playwright install chromium

COPY . .

# Uploads are a working scratch area. The durable copies live in Supabase
# storage, so losing this directory on redeploy is expected, not a data loss.
RUN mkdir -p data/uploads data/out

EXPOSE 8000

# Shell form so $PORT expands. Render supplies it; 8000 is the local default.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
