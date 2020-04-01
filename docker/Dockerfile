FROM pytorch/pytorch:latest
# install
RUN apt-get update --fix-missing
RUN apt-get install -y protobuf-compiler python-pil python-lxml git wget unzip screen vim libgtk2.0-dev

RUN mkdir /app
#COPY ./requirements.txt /tmp/requirements.txt
#COPY . /tmp/
#RUN pip install -r /tmp/requirements.txt
RUN pip install -e .

RUN pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

WORKDIR /app
#ENTRYPOINT ["/bin/bash"]
