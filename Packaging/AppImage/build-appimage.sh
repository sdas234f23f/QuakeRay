#/bin/sh
docker build --tag=build-quakeray docker
docker run --rm --privileged -e VERSION=`./get-version.sh` -v ${PWD}/../..:/usr/src/QuakeRay build-quakeray /usr/src/QuakeRay/Packaging/AppImage/run-in-docker.sh
