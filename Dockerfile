FROM python:3.11.4-bookworm

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
  python3-tk \
  libtcl8.6 \
  libtk8.6

RUN pip install --upgrade pip

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

#CMD ["python3", "./index.py"]
CMD ["bash"]kkj