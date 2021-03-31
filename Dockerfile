FROM python:3.8-slim
EXPOSE 8501

WORKDIR /app
COPY requirements.txt ./requirements.txt

RUN apt-get -y update && apt-get install build-essential cmake pkg-config -y
RUN apt-get -y install python3-opencv

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]
