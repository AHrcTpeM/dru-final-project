FROM python:3.10

WORKDIR /dru-final-project

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

RUN export PYTHONPATH='${PYTHONPATH}:/app'

COPY . .

CMD ["python", "./app.py"]