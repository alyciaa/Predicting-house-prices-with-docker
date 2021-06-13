FROM ubuntu:latest
COPY . /app
RUN apt-get update
RUN set -xe && apt-get update && apt-get install -y python3-pip
RUN pip3 install --upgrade pip
WORKDIR /app
RUN pip3 install -r Requirements.txt
CMD python3 model.py



