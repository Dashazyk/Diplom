# Replace with required container type e.g. base, devel etc in the following line
FROM nvcr.io/nvidia/deepstream:6.2-devel

RUN mkdir -p /diplom/src

WORKDIR /diplom
#ENV GST_LIBS="-lgstreamer-1.0 -lgobject-2.0 -lglib-2.0"

RUN apt -y update
RUN apt -y install -y apt-transport-https ca-certificates -y
RUN update-ca-certificates
RUN apt -y install python-gi-dev python-gi-dev
RUN apt -y install python3-opencv python3-numpy

RUN apt -y install python3-gi python3-dev python3-gst-1.0 python-gi-dev git python-dev \
    python3 python3-pip python3.8-dev cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev

RUN python3 -m pip install deepface
#RUN echo && \
#  git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git && \
#  cd deepstream_python_apps && \
#  git submodule update --init && \
#  cd 3rdparty/gst-python/ && \
#  ./autogen.sh && \
#  make && \
#  make install
#
#RUN cd deepstream_python_apps && \
#    cd bindings &&\
#    mkdir build &&\
#    cd build &&\
#    cmake .. && \
#    make && \
#    pip3 install ./pyds-1.1.6-py3-none*.whl

#RUN cp -R /diplom/deepstream_python_apps/ /opt/nvidia/deepstream/deepstream/sources/

RUN apt -y install libavcodec-dev libavcodec-extra libavcodec-extra58
RUN apt -y install libavutil-dev libavutil56
#RUN apt -y ffmpeg
RUN bash /opt/nvidia/deepstream/deepstream-6.2/user_deepstream_python_apps_install.sh --version 1.1.6
RUN apt -y install vim python-is-python3
RUN apt -y install python3-flask
RUN apt -y install python3-pandas
RUN apt -y install python3-pyaudio
RUN apt -y install python3-pydub

ENV PYTHONPATH=/opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/apps
# To get video driver libraries at runtime (libnvidia-encode.so/libnvcuvid.so)
ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video

COPY src/ /diplom/src
WORKDIR /diplom/src

ENTRYPOINT ["python3", "/diplom/src/diplom_proto.py"]
