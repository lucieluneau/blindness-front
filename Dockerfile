FROM tensorflow/tensorflow:latest

COPY app.py /app.py
COPY requirements-docker.txt /requirements-docker.txt
COPY model /model

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install -r requirements-docker.txt

CMD streamlit run app.py
