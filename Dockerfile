FROM python:3.8-slim

WORKDIR /app/

ARG CONTAINER_NAME
ARG CONNECTION_STRING

ENV CONTAINER_NAME=$CONTAINER_NAME
ENV CONNECTION_STRING=$CONNECTION_STRING

COPY . .

# Update the package lists and install the required packages in one RUN command to keep the image size small
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=app.py

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]