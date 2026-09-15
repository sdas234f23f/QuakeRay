#!/bin/sh
FOLDER=quakeray-${VERSION}_linux64
ARCHIVE=${FOLDER}.tar.gz

cd /usr/src/QuakeRay \
  && rm -rf build/appimage \
  && ./Packaging/AppImage/python3.11.0.AppImage /opt/meson/meson.py build/appimage -Dbuildtype=release -Db_lto=true \
  && ninja -C build/appimage \
  && cd Packaging/AppImage \
  && rm -rf AppDir \
  && rm -rf quakeray* \
  && mkdir ${FOLDER} \
  && ./linuxdeploy-x86_64.AppImage -e ../../build/appimage/quakeray --appdir=AppDir --create-desktop-file \
     -i ../../Misc/QuakeRay_256.png --icon-filename=quakeray --output appimage \
  && cp quakeray-${VERSION}-x86_64.AppImage ${FOLDER}/quakeray.AppImage \
  && cp ../../Quake/vkquake.pak ${FOLDER} \
  && cp ../../LICENSE.txt ${FOLDER} \
  && tar -zcvf ${ARCHIVE} ${FOLDER}
