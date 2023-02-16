# first stage
FROM python:3.9 AS builder
COPY ./requirements.txt .

RUN pip install --user --upgrade pip && pip install --user --no-cache-dir -r requirements.txt

# second unnamed stage
FROM python:3.9-slim

# set work directory
WORKDIR /app

# copy only the dependencies installation from the 1st stage image
COPY --from=builder /root/.local /root/.local

# install project
COPY . .
RUN pip install .

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBUG 0

# add and run as non-root user
RUN useradd --create-home --shell /bin/bash myuser
USER myuser

# run uvicorn
CMD uvicorn startleiter.app:app --host=0.0.0.0 --port=$PORT
