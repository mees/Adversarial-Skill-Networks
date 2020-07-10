FROM pytorch/pytorch:latest
# install
RUN apt-get update --fix-missing
RUN apt-get install -y protobuf-compiler python-pil python-lxml git wget unzip screen vim libgtk2.0-dev

#COPY ./requirements.txt /tmp/requirements.txt
COPY . /tmp/
#RUN pip install -r /tmp/requirements.txt
#WORKDIR /app
#RUN pip install -e /tmp/.

RUN ls
RUN pip install -e .

WORKDIR /app
#RUN pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

#ENTRYPOINT ["/bin/bash"]
