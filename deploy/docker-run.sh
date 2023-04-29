docker container prune -f
#docker run --gpus=all -it --rm --net=host --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -w /opt/nvidia/deepstream/deepstream-6.2 --name deploy-nvidiploma-1 ournvidia:latest
docker run --gpus all \
	-it --rm            \
	--net=host          \
	--privileged        \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY=$DISPLAY              \
	--mount type=bind,source="$(pwd)/../src/"/,target=/diplom/src \
	--mount type=bind,source="$(pwd)/../yolo/"/,target=/diplom/yolo \
	--mount type=bind,source="$(pwd)/../face_db/"/,target=/diplom/face_db \
	--mount type=bind,source="/tmp/face_from_frame_storage/"/,target=/tmp/face_from_frame_storage \
	--name deploy-nvidiploma-1 ournvidia:latest
#nvcr.io/nvidia/deepstream:6.2-devel
