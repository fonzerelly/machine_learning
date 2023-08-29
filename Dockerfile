FROM python:3.11.4-bookworm

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
  python3-tk \
  libtcl8.6 \
  libtk8.6

RUN pip install --upgrade pip

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-mnist
RUN mkdir /usr/scripts && \
    echo "export PATH=$PATH:/usr/scripts" >> /root/.bashrc && \
    echo "mnist_preview --data /usr/src/mnist --id \$1" >> /usr/scripts/mnist.sh && \
    chmod 777 /usr/scripts/mnist.sh

COPY ./mnist /usr/src/mnist

#CMD ["python3", "./index.py"]
CMD ["bash"]