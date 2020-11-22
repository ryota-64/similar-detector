FROM pymesh/pymesh
RUN apt-get update && apt-get install -y libopencv-dev python3-opencv
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

