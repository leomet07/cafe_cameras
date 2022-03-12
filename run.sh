xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
docker run -m 8GB -it --rm -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH -v $PWD:/project -w /project -it ilya_project_image python main.py
# docker run -u $UID:$UID -v $PWD:/project -w /project --runtime=nvidia --init --rm -it tensorflow/tensorflow:1.14.0-gpu-py3 bash
xhost -local:docker
