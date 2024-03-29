#comes with python + git and some other utilities
FROM ubuntu:20.04 AS base

#https://stackoverflow.com/a/25908200/5228524
#https://github.com/CircleCI-Public/circleci-dockerfiles/blob/eda702fe560b258138d3a895c5d397f4a2cdfcd0/python/images/3.6.12/Dockerfile#L83
#https://dev.to/emmanuelnk/using-sudo-without-password-prompt-as-non-root-docker-user-52bg
RUN apt-get update && \
      apt-get -y install sudo adduser

RUN adduser --disabled-password --gecos '' astropath && \
    adduser astropath sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Install PowerShell From microsoft doc: https://docs.microsoft.com/en-us/powershell/scripting/install/install-ubuntu?view=powershell-7.2
Run sudo apt update
RUN sudo apt-get install -y wget apt-transport-https software-properties-common
RUN wget -q https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
# Register the Microsoft repository GPG keys
RUN sudo dpkg -i packages-microsoft-prod.deb
# Update the list of packages after we added packages.microsoft.com
RUN sudo apt-get update
# Install PowerShell
RUN sudo apt-get install -y powershell

# mono runtime installation for Ubuntu 20.04 from here:
# https://linuxize.com/post/how-to-install-mono-on-ubuntu-20-04/
RUN sudo apt-get update && \
    sudo apt-get upgrade -y && \
    # Install pre-requisite packages.
    sudo apt-get install -y --no-install-recommends dirmngr gnupg apt-transport-https ca-certificates software-properties-common && \
    # Import the repository’s GPG key
    sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF && \
    # Add the Mono repository to the system sources’ list
    sudo apt-add-repository 'deb https://download.mono-project.com/repo/ubuntu stable-focal main' && \
    # Install Mono
    sudo apt-get install -y --no-install-recommends mono-runtime && \
    # Remove dependencies
    sudo apt-get purge -y dirmngr gnupg apt-transport-https software-properties-common && \
    sudo apt-get autoremove -y

#libvips stuff is from https://github.com/TailorBrands/docker-libvips/blob/f64952af6871fb963934b7454f7edf5b6738f4b8/8.6.1/Dockerfile
#gdal stuff is from https://github.com/OSGeo/gdal/blob/f6689b653338502de4150f491864553894c37410/gdal/docker/ubuntu-full/Dockerfile

#make the home directory editable
RUN sudo mkdir -p /home/astropath/.cache/pip /home/astropath/.cache/matplotlib
RUN sudo chmod -R a+w /home/astropath

RUN sudo apt-get update && \
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  #for gdal
  ant \
  #for vips
  automake \
  bison \
  build-essential \
  ccache \
  cdbs \
  cmake \
  curl \
  debhelper \
  #for gdal
  default-libmysqlclient-dev \
  #for vips
  dh-autoreconf \
  #for build
  equivs \
  #for vips
  fftw3-dev \
  flex \
  gettext \
  #for build
  git \
  #for vips
  gtk-doc-tools \
  #for gdal
  libarmadillo-dev \
  libboost-dev \
  libcairo2-dev \
  #for vips and gdal
  libcfitsio-dev \
  #for gdal
  libcharls-dev \
  libcrypto++-dev \
  libdap-dev \
  libdeflate-dev \
  libepsilon-dev \
  libfyba-dev \
  #for vips
  libexif-gtk-dev \
  #for gdal
  libexpat-dev \
  #for vips
  libfontconfig1-dev \
  libfreetype6-dev \
  #for gdal
  libfreexl-dev \
  libgeos-dev \
  #for vips and gdal
  libgif-dev \
  #for vips
  libgirepository1.0-dev \
  libglib2.0-dev \
  libgsf-1-dev \
  #for gdal
  libhdf4-alt-dev \
  libhdf5-serial-dev \
  libheif-dev \
  #for vips
  libice-dev \
  #for vips and gdal
  libjpeg-dev \
  #for gdal
  libkml-dev \
  liblcms2-2 \
  #for vips
  liblcms2-dev \
  #for gdal
  liblzma-dev \
  #for vips
  libmagickwand-dev \
  libmatio-dev \
  #for gdal
  libnetcdf-dev \
  #for vips and gdal
  libopenexr-dev \
  #for gdal
  libopenjp2-7-dev \
  #for vips
  libopenslide-dev \
  liborc-0.4-dev \
  libpango1.0-dev \
  #for gdal
  libpcre3-dev \
  #for vips and gdal
  libpng-dev \
  #for gdal
  libpoppler-dev \
  #for vips
  libpoppler-glib-dev \
  #for gdal
  libpoppler-private-dev \
  libpq-dev \
  libpython3-dev \
  #for vips
  librsvg2-dev \
  #for gdal
  libspatialite-dev \
  libsqlite3-dev \
  libssl-dev \
  #for vips
  libtiff-dev \
  #for vips and gdal
  libwebp-dev \
  #for gdal
  libxerces-c-dev \
  #for vips
  libxml-parser-perl \
  #for vips and gdal
  libxml2-dev \
  #for gdal
  libzstd-dev \
  #for vips
  pkg-config \
  #for astropath
  python-is-python3 \
  python3-pip \
  #for gdal
  rsync \
  sqlite3 \
  swig \
  unixodbc-dev \
  unzip \
  wget \
  zip \
  #for vips
  zlib1g-dev

#for gdal
ARG JAVA_VERSION=11
RUN sudo DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-"$JAVA_VERSION"-jdk

RUN sudo apt-get autoclean && \
    sudo apt-get autoremove && \
    sudo rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

from base as libvips

ENV LIBVIPS_VERSION_MAJOR 8
ENV LIBVIPS_VERSION_MINOR 12
ENV LIBVIPS_VERSION_PATCH 1
ENV LIBVIPS_VERSION $LIBVIPS_VERSION_MAJOR.$LIBVIPS_VERSION_MINOR.$LIBVIPS_VERSION_PATCH

RUN \
  # Build libvips
  cd /tmp && \
  curl -L -O https://github.com/libvips/libvips/releases/download/v$LIBVIPS_VERSION/vips-$LIBVIPS_VERSION.tar.gz && \
  tar zxvf vips-$LIBVIPS_VERSION.tar.gz
RUN \
  cd /tmp/vips-$LIBVIPS_VERSION && \
  ./configure --enable-debug=no --without-python $1
RUN \
  cd /tmp/vips-$LIBVIPS_VERSION && \
  make
RUN \
  cd /tmp/vips-$LIBVIPS_VERSION && \
  sudo make install
RUN \
  cd /tmp/vips-$LIBVIPS_VERSION && \
  sudo ldconfig

from base as gdal

# Build likbkea
ARG KEA_VERSION=1.4.13
RUN sudo wget https://github.com/ubarsc/kealib/archive/kealib-${KEA_VERSION}.zip \
    && sudo unzip kealib-${KEA_VERSION}.zip \
    && sudo rm -f kealib-${KEA_VERSION}.zip \
    && cd kealib-kealib-${KEA_VERSION} \
    && sudo cmake . -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr -DHDF5_INCLUDE_DIR=/usr/include/hdf5/serial \
        -DHDF5_LIB_PATH=/usr/lib/x86_64-linux-gnu/hdf5/serial -DLIBKEA_WITH_GDAL=OFF \
    && sudo make -j$(nproc) \
    && sudo make install DESTDIR="/build_thirdparty" \
    && sudo make install \
    && cd .. \
    && sudo rm -rf kealib-kealib-${KEA_VERSION} \
    && for i in /build_thirdparty/usr/lib/*; do sudo strip -s $i 2>/dev/null || /bin/true; done \
    && for i in /build_thirdparty/usr/bin/*; do sudo strip -s $i 2>/dev/null || /bin/true; done

# Build mongo-c-driver
ARG MONGO_C_DRIVER_VERSION=1.16.2
RUN sudo mkdir mongo-c-driver \
    && wget -q https://github.com/mongodb/mongo-c-driver/releases/download/${MONGO_C_DRIVER_VERSION}/mongo-c-driver-${MONGO_C_DRIVER_VERSION}.tar.gz -O - \
        | sudo tar xz -C mongo-c-driver --strip-components=1 \
    && cd mongo-c-driver \
    && sudo mkdir build_cmake \
    && cd build_cmake \
    && sudo cmake .. -DCMAKE_INSTALL_PREFIX=/usr -DENABLE_TESTS=NO -DCMAKE_BUILD_TYPE=Release \
    && sudo make -j$(nproc) \
    && sudo make install DESTDIR="/build_thirdparty" \
    && sudo make install \
    && cd ../.. \
    && sudo rm -rf mongo-c-driver \
    && sudo rm /build_thirdparty/usr/lib/x86_64-linux-gnu/*.a \
    && for i in /build_thirdparty/usr/lib/x86_64-linux-gnu/*; do sudo strip -s $i 2>/dev/null || /bin/true; done \
    && for i in /build_thirdparty/usr/bin/*; do sudo strip -s $i 2>/dev/null || /bin/true; done

# Build mongocxx
ARG MONGOCXX_VERSION=3.5.0
RUN sudo mkdir mongocxx \
    && wget -q https://github.com/mongodb/mongo-cxx-driver/archive/r${MONGOCXX_VERSION}.tar.gz -O - \
        | sudo tar xz -C mongocxx --strip-components=1 \
    && cd mongocxx \
    && sudo mkdir build_cmake \
    && cd build_cmake \
    && sudo cmake .. -DCMAKE_INSTALL_PREFIX=/usr -DBSONCXX_POLY_USE_BOOST=ON -DMONGOCXX_ENABLE_SLOW_TESTS=NO -DCMAKE_BUILD_TYPE=Release -DBUILD_VERSION=${MONGOCXX_VERSION} \
    && sudo make -j$(nproc) \
    && sudo make install DESTDIR="/build_thirdparty" \
    && sudo make install \
    && cd ../.. \
    && sudo rm -rf mongocxx \
    && for i in /build_thirdparty/usr/lib/x86_64-linux-gnu/*; do sudo strip -s $i 2>/dev/null || /bin/true; done \
    && for i in /build_thirdparty/usr/bin/*; do sudo strip -s $i 2>/dev/null || /bin/true; done

# Build tiledb
ARG TILEDB_VERSION=2.0.8
RUN sudo mkdir tiledb \
    && wget -q https://github.com/TileDB-Inc/TileDB/archive/${TILEDB_VERSION}.tar.gz -O - \
        | sudo tar xz -C tiledb --strip-components=1 \
    && cd tiledb \
    && sudo mkdir build_cmake \
    && cd build_cmake \
    && sudo ../bootstrap --prefix=/usr \
    && sudo make -j$(nproc) \
    && sudo make install-tiledb DESTDIR="/build_thirdparty" \
    && sudo make install-tiledb \
    && cd ../.. \
    && sudo rm -rf tiledb \
    && for i in /build_thirdparty/usr/lib/x86_64-linux-gnu/*; do sudo strip -s $i 2>/dev/null || /bin/true; done \
    && for i in /build_thirdparty/usr/bin/*; do sudo strip -s $i 2>/dev/null || /bin/true; done

# Build openjpeg
ARG OPENJPEG_VERSION=
RUN if test "${OPENJPEG_VERSION}" != ""; then ( \
    sudo wget -q https://github.com/uclouvain/openjpeg/archive/v${OPENJPEG_VERSION}.tar.gz \
    && sudo tar xzf v${OPENJPEG_VERSION}.tar.gz \
    && sudo rm -f v${OPENJPEG_VERSION}.tar.gz \
    && cd openjpeg-${OPENJPEG_VERSION} \
    && sudo cmake . -DBUILD_SHARED_LIBS=ON  -DBUILD_STATIC_LIBS=OFF -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
    && sudo make -j$(nproc) \
    && sudo make install \
    && sudo mkdir -p /build_thirdparty/usr/lib/x86_64-linux-gnu \
    && sudo rm -f /usr/lib/x86_64-linux-gnu/libopenjp2.so* \
    && sudo mv /usr/lib/libopenjp2.so* /usr/lib/x86_64-linux-gnu \
    && sudo cp -P /usr/lib/x86_64-linux-gnu/libopenjp2.so* /build_thirdparty/usr/lib/x86_64-linux-gnu \
    && for i in /build_thirdparty/usr/lib/x86_64-linux-gnu/*; do sudo strip -s $i 2>/dev/null || /bin/true; done \
    && cd .. \
    && sudo rm -rf openjpeg-${OPENJPEG_VERSION} \
    ); fi

# Install MDB Driver Jars
RUN sudo wget -q https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/mdb-sqlite/mdb-sqlite-1.0.2.tar.bz2 \
  && sudo tar -xjf mdb-sqlite-1.0.2.tar.bz2 \
  && sudo mkdir -p /build/usr/share/java \
  && sudo cp mdb-sqlite-1.0.2/lib/*.jar /build/usr/share/java \
  && sudo rm -rf mdb-sqlite-1.0.2.tar.bz2 && sudo rm -rf mdb-sqlite-1.0.2

#Set CLASSPATH so jars are found
ENV CLASSPATH="/build/usr/share/java/jackcess-1.1.14.jar:/build/usr/share/java/commons-logging-1.1.1.jar:/build/usr/share/java/commons-lang-2.4.jar"

#Build File Geodatabase

ARG WITH_FILEGDB=
RUN if echo "$WITH_FILEGDB" | grep -Eiq "^(y(es)?|1|true)$"  ; then ( \
  sudo wget -q https://github.com/Esri/file-geodatabase-api/raw/master/FileGDB_API_1.5.1/FileGDB_API_1_5_1-64gcc51.tar.gz \
  && sudo tar -xzf FileGDB_API_1_5_1-64gcc51.tar.gz \
  && sudo chown -R root:root FileGDB_API-64gcc51 \
  && sudo mv FileGDB_API-64gcc51 /usr/local/FileGDB_API \
  && sudo rm -rf /usr/local/FileGDB_API/lib/libstdc++* \
  && sudo cp /usr/local/FileGDB_API/lib/* /build_thirdparty/usr/lib \
  && sudo cp /usr/local/FileGDB_API/include/* /usr/include \
  && sudo rm -rf FileGDB_API_1_5_1-64gcc51.tar.gz \
  ) ; fi

ARG RSYNC_REMOTE

ARG WITH_DEBUG_SYMBOLS=no

# Build PROJ
ARG PROJ_VERSION=8.0.1
ARG PROJ_INSTALL_PREFIX=/usr/local
COPY ./bh-proj.sh /buildscripts/bh-proj.sh
RUN sudo -E bash /buildscripts/bh-proj.sh

# Build GDAL
ARG JAVA_VERSION=11
ARG GDAL_VERSION=v3.3.0
ARG GDAL_RELEASE_DATE
ARG GDAL_BUILD_IS_RELEASE
ENV LD_LIBRARY_PATH=/build/usr/local/lib/
COPY ./bh-gdal.sh /buildscripts/bh-gdal.sh
RUN sudo -E bash /buildscripts/bh-gdal.sh

FROM base as python_packages

COPY --from=gdal  /build_thirdparty/usr/ /usr/
ARG PROJ_INSTALL_PREFIX=/usr/local
COPY --from=gdal  /build${PROJ_INSTALL_PREFIX}/share/proj/ ${PROJ_INSTALL_PREFIX}/share/proj/
COPY --from=gdal  /build${PROJ_INSTALL_PREFIX}/include/ ${PROJ_INSTALL_PREFIX}/include/
COPY --from=gdal  /build${PROJ_INSTALL_PREFIX}/bin/ ${PROJ_INSTALL_PREFIX}/bin/
COPY --from=gdal  /build${PROJ_INSTALL_PREFIX}/lib/ ${PROJ_INSTALL_PREFIX}/lib/
COPY --from=gdal  /build/usr/share/java /usr/share/java
COPY --from=gdal  /build/usr/share/gdal/ /usr/share/gdal/
COPY --from=gdal  /build/usr/include/ /usr/include/
COPY --from=gdal  /build_gdal_python/usr/ /usr/
COPY --from=gdal  /build_gdal_version_changing/usr/ /usr/
RUN sudo ldconfig

#add python packages
ARG GITHUB_TOKEN
ARG ASTROPATH_DEPENDENCY_COMMIT=50aa7189e1999d406588acc1ae565233eafa8779

RUN sudo pip3 install --upgrade pip 'setuptools<66' packaging

RUN \
  sudo git config --global url."https://astropathjhujenkins:${GITHUB_TOKEN}@github".insteadOf https://github && \
  sudo git clone https://github.com/AstropathJHU/AstroPathPipelinePrivate && \
  cd AstroPathPipelinePrivate && \
  sudo git fetch && \
  sudo git checkout "${ASTROPATH_DEPENDENCY_COMMIT}" && \
  sudo git config --global --unset url."https://astropathjhujenkins:${GITHUB_TOKEN}@github".insteadOf

ARG ASTROPATH_INSTALL_EXTRAS=true

RUN sudo pip3 install --use-pep517 pyvips
RUN sudo pip3 install git+https://github.com/MIC-DKFZ/nnUNet.git@refs/pull/1209/head
RUN \
  if $ASTROPATH_INSTALL_EXTRAS; then \
    EXTRAS="[all]"; \
  else \
    EXTRAS=; \
  fi && \
  sudo pip3 install "./AstroPathPipelinePrivate$EXTRAS"
RUN sudo pip3 uninstall -y astropath
RUN chown -R root:root /usr/local/lib/python3.8/

from base as final

copy --from=libvips /usr/local/bin/*vips* /usr/local/bin/
copy --from=libvips /usr/local/lib/*vips* /usr/local/lib/
COPY --from=gdal  /build_thirdparty/usr/ /usr/
ARG PROJ_INSTALL_PREFIX=/usr/local
COPY --from=gdal  /build${PROJ_INSTALL_PREFIX}/share/proj/ ${PROJ_INSTALL_PREFIX}/share/proj/
COPY --from=gdal  /build${PROJ_INSTALL_PREFIX}/include/ ${PROJ_INSTALL_PREFIX}/include/
COPY --from=gdal  /build${PROJ_INSTALL_PREFIX}/bin/ ${PROJ_INSTALL_PREFIX}/bin/
COPY --from=gdal  /build${PROJ_INSTALL_PREFIX}/lib/ ${PROJ_INSTALL_PREFIX}/lib/
copy --from=python_packages /usr/local/lib/python3.8/dist-packages/ /usr/local/lib/python3.8/dist-packages/
RUN sudo ldconfig
#RUN sudo projsync --system-directory --all

COPY --from=gdal  /build/usr/share/java /usr/share/java
COPY --from=gdal  /build/usr/share/gdal/ /usr/share/gdal/
COPY --from=gdal  /build/usr/include/ /usr/include/
COPY --from=gdal  /build_gdal_python/usr/ /usr/
COPY --from=gdal  /build_gdal_version_changing/usr/ /usr/

RUN sudo ldconfig

ENV CLASSPATH="/usr/share/java/jackcess-1.1.14.jar:/usr/share/java/commons-logging-1.1.1.jar:/usr/share/java/commons-lang-2.4.jar"
ENV PATH=/home/astropath/.local/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/lib/:${LD_LIBRARY_PATH}

USER astropath
LABEL astropath=true

CMD ["/bin/bash"]
