FROM tensorflow/tensorflow:1.14.0-gpu-py3

# Install system packages
ENV TZ=US/Eastern

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget \
      python3-tk && \
    rm -rf /var/lib/apt/lists/*
    
# Setting up working directory 
RUN mkdir /project
WORKDIR /project

# COPY requirements.txt requirements.txt

RUN pip install --upgrade pip

RUN pip install --no-cache-dir opencv-python

# Minimize image size 
RUN (apt-get autoremove -y; \
     apt-get autoclean -y)  

ENV QT_X11_NO_MITSHM=1

CMD ["bash"]
#CMD ["python", "show.py"]
