ARG IMAGE
FROM $IMAGE

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y ffmpeg

RUN pip3 install -U torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html -U

RUN pip3 install mmdet==2.22.0
RUN pip3 install mmcls==0.23.1
RUN pip3 install pyyaml==6.0
RUN pip3 install scikit-learn==1.0.2

RUN mkdir -p /tmp/mmdet \
    && wget https://github.com/open-mmlab/mmdetection/archive/refs/tags/v2.22.0.zip -P /tmp/mmdet

RUN apt-get update && apt-get install -y openssh-server
EXPOSE 22

RUN apt-get install -y sudo
RUN mkdir -p /run/sshd

ARG home=/root
RUN mkdir $home/.ssh
COPY my_key.pub $home/.ssh/authorized_keys
RUN chown root:root $home/.ssh/authorized_keys && \
    chmod 600 $home/.ssh/authorized_keys

COPY sshd_daemon.sh /sshd_daemon.sh
RUN chmod 755 /sshd_daemon.sh
CMD ["/sshd_daemon.sh"]
ENTRYPOINT ["sh", "-c", "/sshd_daemon.sh"]