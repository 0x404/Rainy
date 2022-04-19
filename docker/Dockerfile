FROM python:3.8

ARG workdir=/opt/rainy
RUN mkdir -p ${workdir}
WORKDIR ${workdir}

RUN pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY . .
RUN pip3 install -r docker/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

CMD [ "/bin/bash" ]
