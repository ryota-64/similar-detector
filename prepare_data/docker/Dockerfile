FROM pymesh/pymesh
RUN apt-get update && apt-get install -y libopencv-dev python3-opencv
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
# ユーザーを作成
ARG UID=1000
RUN useradd -m -u ${UID} prepare

# 作成したユーザーに切り替える
USER ${UID}
WORKDIR /prepare

ENV PYTHONPATH ${PYTHONPATH}:/prepare/similar-detector/
