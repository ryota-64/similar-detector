FROM pytorch/pytorch
RUN apt-get update && apt-get install -y tzdata && apt-get install -y libopencv-dev
RUN apt install -y jupyter-notebook
# ユーザーを作成
ARG UID=1000
RUN useradd -m -u ${UID} detector

# 作成したユーザーに切り替える
USER ${UID}
COPY requirements.txt /workspace/
RUN pip install -r requirements.txt
RUN pip install Cython
RUN pip install libmr
WORKDIR /workspace/similar-detector
ENV PATH ${PATH}:/home/detector/.local/bin/
ENV PYTHONPATH ${PYTHONPATH}:/workspace/similar-detector/
#COPY libMR/ /workspace/libMR
#WORKDIR /workspace/libMR
#RUN ["/bin/bash", "-c", "./compile.sh"]


