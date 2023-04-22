docker container prune -f
docker images prune -f
cd ..
docker build -t 'ournvidia:latest' . -f ./deploy/Dockerfile

