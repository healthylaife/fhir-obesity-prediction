FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

ENTRYPOINT ["python","./main.py" ]