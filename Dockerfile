FROM tensorflow/tensorflow

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

CMD ["python", "some_python_file.py"]
