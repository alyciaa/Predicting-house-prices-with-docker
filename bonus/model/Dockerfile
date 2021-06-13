FROM ubuntu:latest
COPY . /app
WORKDIR /app
RUN apt-get update
RUN set -xe && apt-get update && apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install -r Requirements.txt
CMD python3 model.py




