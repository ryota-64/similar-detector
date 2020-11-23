FROM pytorch/pytorch
RUN apt-get update && apt-get install -y tzdata && apt-get install -y libopencv-dev
COPY requirements.txt /workspace/
RUN pip install -r requirements.txt
RUN pip install Cython
RUN pip install libmr
#COPY libMR/ /workspace/libMR
#WORKDIR /workspace/libMR
#RUN ["/bin/bash", "-c", "./compile.sh"]

# ユーザーを作成
ARG UID=1000
RUN useradd -m -u ${UID} detector

# 作成したユーザーに切り替える
USER ${UID}
WORKDIR /workspace
