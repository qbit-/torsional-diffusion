#!/bin/bash

usage() {
  echo "(no arguments) - just cleans old containers with default name, run with container name "\
       "${USER}_dev_{SERVICE_NAME} and with no ports exposed. Do not forget to make container down if you don't plan "\
       "to use it further. Keyword arguments should be placed after flags h, d, l."
  echo "-h (help) - print this message."
  echo "-d (down) - just stop and remove container with given name [optional], remove hanging connections."
  echo "-y (dry-run) - dry-run of restart_toolkit.sh -d command. Shows containers to remove."
  echo "-l (list) - list all the containers of the current user (existing containers containing "\
       "${USER}_dev_{SERVICE_NAME} int its name."
  echo "-n (name) - prefix of the name of the toolkit container to stop, remove and restart: " \
       "${NAME}_${USER}_dev_{SERVICE_NAME}"
  echo "-p (port) - expose a port from the toolkit container. Use the same port on the host machine for binding, like " \
       "docker run -p <port>:<port>"
  echo "-v (volume) - specify UP TO ONE additional volume binding. Will be used as docker run -v <binding>, so use "\
       "string with syntax -v \"host_folder:container_folder\", do not forget to quote the string."
  echo "-o (host machine port) - provide an optional port for the container on the host machine. Example: "\
       "docker run -p <host machine port>:<port>"
}


while getopts hdylv:p:n:o: opt; do
  case "${opt}" in
  h)
    usage
    exit
    ;;
  d)
    DOWN="SET"
    ;;
  y)
    DRY_RUN="SET"
    ;;
  l)
    LIST_CONTAINERS="SET"
    ;;
  p)
    PORT=$OPTARG
    ;;
  v)
    VOLUME_SET=$OPTARG
    ;;
  o)
    HOST_PORT=$OPTARG
    ;;
  n)
    NAME=$OPTARG
    ;;
  *)
    echo "illegal option"
    usage
    exit 1
    ;;
  esac
done

set -x

if [ -z ${NAME+x} ]; then
  NAME=${USER}_torsional_diffusion
else
  NAME=${NAME}_${USER}_torsional_diffusion
fi

NAME=$(echo $NAME | sed -r 's/\.//g')

echo "--> Using grep with container names prefix:"
echo "=== ${NAME} ==="

if [ ! -z ${LIST_CONTAINERS+x} ]; then
  docker ps -a | grep $NAME
  exit
fi

if [ ! -z ${DRY_RUN+x} ]; then
  echo "--> Would stop containers:"
  docker container  ls -a | grep -E "\b$NAME"
  echo "--> Would remove stopped containers:"
  docker container  ls -a | grep -E "\b$NAME"
  echo "--> Would remove orphan networks:"
  docker network ls | grep -E "\b"$NAME
  exit
fi

echo "--> Stopping containers"
docker stop $(docker container  ls -a | grep -E "\b$NAME" | awk '{print $1}')
echo "--> Removing stopped containers"
docker rm $(docker container  ls -a | grep -E "\b$NAME" | awk '{print $1}')
echo "--> Removing orphan networks"
docker network rm $(docker network ls | grep -E "\b"$NAME"" | awk '{print $1}')

echo "--> Containers down"

if [ ! -z ${DOWN+x} ]; then
  exit
fi

if [ ! -z ${VOLUME_SET+x} ]; then
  VOLUME_SET="--volume="$VOLUME_SET
else
  VOLUME_SET=""
fi

# Increase HTTP timeout for docker compose
export COMPOSE_HTTP_TIMEOUT=200

if [ -z ${PORT+x} ]; then
  docker-compose --verbose -f ./docker/docker-compose.yml -p $NAME run -d $VOLUME_SET toolkit
else
  if [ -z ${HOST_PORT+x} ]; then
    docker-compose --verbose -f ./docker/docker-compose.yml -p $NAME run -d -p $PORT:$PORT $VOLUME_SET toolkit
  else
    docker-compose --verbose -f ./docker/docker-compose.yml -p $NAME run -d -p $HOST_PORT:$PORT \
                   $VOLUME_SET \
                   toolkit
  fi
fi
