FROM ubuntu:16.04

WORKDIR /tensor

ADD . /tensor

RUN apt-get update

RUN apt-get -y install python3

RUN apt-get -y install python3-pip

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

RUN pip3 install jupyter 

EXPOSE 5000

CMD [ "python3", "app.py" ] 
