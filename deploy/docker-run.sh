docker container prune -f
#docker run --gpus=all -it --rm --net=host --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -w /opt/nvidia/deepstream/deepstream-6.2 --name deploy-nvidiploma-1 ournvidia:latest
docker run --gpus all \
	-it --rm            \
	--net=host          \
	--privileged        \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY=$DISPLAY              \
	--name deploy-nvidiploma-1 ournvidia:latest
#nvcr.io/nvidia/deepstream:6.2-devel
