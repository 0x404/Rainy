FROM python:3.8

ARG workdir=/opt/rainy
RUN mkdir -p ${workdir}
WORKDIR ${workdir}

RUN pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN sed -i s@/deb.debian.org/@/mirrors.aliyun.com/@g /etc/apt/sources.list\
    && apt-get clean\
    && apt-get update\
    && apt-get install -y vim\
    && apt-get install -y git
COPY . .
RUN pip3 install -r docker/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

CMD [ "/bin/bash" ]
