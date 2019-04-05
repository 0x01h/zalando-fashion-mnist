docker build .
docker run -it -p 8888:8888 $(docker images | awk '{print $3}' | awk 'NR==2')