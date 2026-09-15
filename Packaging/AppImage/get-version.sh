#!/bin/sh
BASE_VERSION=`grep -oP "^#define\s*ENGINE_VERSION\s*\K[0-9.]*" ../../Quake/quakedef.h`
PATCH_VERSION=`grep -oP "^#define\s*ENGINE_VER_PATCH\s*\K[0-9.]*" ../../Quake/quakedef.h`
SUFFIX=`grep -oP "^#define\s*ENGINE_VER_SUFFIX\s*\"\K([^\"]*)" ../../Quake/quakedef.h`
echo ${BASE_VERSION}.${PATCH_VERSION}${SUFFIX}
