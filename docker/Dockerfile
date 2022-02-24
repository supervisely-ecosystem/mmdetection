FROM supervisely/base-py-sdk:6.4.3

RUN pip3 install -U torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install mmcv-full==1.4.4 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html -U

RUN pip3 install mmdet==2.21.0
RUN pip3 install pyyaml==6.0

RUN mkdir -p /tmp/mmdet \
    && wget https://github.com/open-mmlab/mmdetection/archive/refs/tags/v2.21.0.tar.gz -P /tmp/mmdet \
    && tar -xvf /tmp/mmdet/v2.21.0.tar.gz -C /tmp/mmdet
