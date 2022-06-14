# pull official base image
FROM python:3.9-slim

# set work directory
WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBUG 0

RUN apt update \
    && apt install --no-install-recommends -y build-essential gcc ca-certificates \
    && apt clean && rm -rf /var/lib/apt/lists/*

# install dependencies
COPY ./requirements.txt .
# RUN pip install --upgrade pip && pip install tensorflow -f https://tf.kmtea.eu/whl/stable.html && pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# install project
COPY . .
RUN pip install .

# add and run as non-root user
RUN useradd --create-home --shell /bin/bash myuser
USER myuser

# run uvicorn
CMD uvicorn startleiter.app:app --host=0.0.0.0 --port=$PORT
