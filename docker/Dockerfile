FROM supervisely/base-py-sdk:6.62.0

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y ffmpeg

RUN pip3 install -U torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html -U

RUN pip3 install mmdet==2.22.0
RUN pip3 install mmcls==0.23.1
RUN pip3 install pyyaml==6.0
RUN pip3 install scikit-learn==1.0.2
RUN pip3 install yapf==0.40.1

RUN pip3 install ffmpeg-python==0.2.0
RUN pip3 install imagecorruptions==1.1.2

RUN mkdir -p /tmp/mmdet \
&& wget https://github.com/open-mmlab/mmdetection/archive/refs/tags/v2.22.0.zip -P /tmp/mmdet

RUN pip3 install supervisely==6.73.347

LABEL python_sdk_version=6.73.347
