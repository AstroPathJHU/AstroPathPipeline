#comes with python + git and some other utilities
FROM circleci/python:3.6.12 AS base

#libvips stuff is from https://github.com/TailorBrands/docker-libvips/blob/f64952af6871fb963934b7454f7edf5b6738f4b8/8.6.1/Dockerfile

#make the home directory editable
RUN sudo mkdir -p /home/circleci/.cache/pip /home/circleci/.cache/matplotlib
RUN sudo chmod -R a+w /home/circleci

#apt-get stuff
RUN sudo apt-get install equivs

FROM base as texlive

#install texlive
ENV PATH=/usr/local/texlive/bin/x86_64-linux:${PATH}
ADD .dockerstuff/texlive-profile.txt /tmp/texlive-profile.txt
ADD .dockerstuff/debian-equivs.txt /tmp/debian-equivs.txt

RUN cd /tmp && \
    wget http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz && \
    mkdir /tmp/install-tl && \
    tar -xzf /tmp/install-tl-unx.tar.gz -C /tmp/install-tl --strip-components=1 && \
    sudo /tmp/install-tl/install-tl --profile=/tmp/texlive-profile.txt --repository=http://mirror.its.dal.ca/ctan/systems/texlive/tlnet

RUN cd /tmp && \
    sudo equivs-control texlive-local && \
    sudo equivs-build debian-equivs.txt && \
    sudo dpkg -i texlive-local*.deb && \
    sudo apt-get install -f && \
    sudo apt-get autoclean && \
    sudo apt-get autoremove && \
    sudo rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

from base as libvips

RUN sudo apt-get update && \
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  automake build-essential curl \
  cdbs debhelper dh-autoreconf flex bison \
  libjpeg-dev libtiff-dev libpng-dev libgif-dev librsvg2-dev libpoppler-glib-dev zlib1g-dev fftw3-dev liblcms2-dev \
  liblcms2-dev libmagickwand-dev libfreetype6-dev libpango1.0-dev libfontconfig1-dev libglib2.0-dev libice-dev \
  gettext pkg-config libxml-parser-perl libexif-gtk-dev liborc-0.4-dev libopenexr-dev libmatio-dev libxml2-dev \
  libcfitsio-dev libopenslide-dev libwebp-dev libgsf-1-dev libgirepository1.0-dev gtk-doc-tools

ENV LIBVIPS_VERSION_MAJOR 8
ENV LIBVIPS_VERSION_MINOR 6
ENV LIBVIPS_VERSION_PATCH 1
ENV LIBVIPS_VERSION $LIBVIPS_VERSION_MAJOR.$LIBVIPS_VERSION_MINOR.$LIBVIPS_VERSION_PATCH

RUN \
  # Build libvips
  cd /tmp && \
  curl -L -O https://github.com/jcupitt/libvips/releases/download/v$LIBVIPS_VERSION/vips-$LIBVIPS_VERSION.tar.gz && \
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

RUN \
  # Clean up
  sudo apt-get remove -y automake curl build-essential && \
  sudo apt-get autoremove -y && \
  sudo apt-get autoclean && \
  sudo apt-get clean && \
  sudo rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#if you add more latex packages, it will build faster
#if you add them all the way at the bottom, because it can use
#the cache from previous builds

from texlive as latex_packages

#add latex packages
RUN sudo /usr/local/texlive/bin/x86_64-linux/tlmgr update --self
RUN sudo /usr/local/texlive/bin/x86_64-linux/tlmgr install caption
RUN sudo /usr/local/texlive/bin/x86_64-linux/tlmgr install cleveref
RUN sudo /usr/local/texlive/bin/x86_64-linux/tlmgr install cm-super
RUN sudo /usr/local/texlive/bin/x86_64-linux/tlmgr install dvipng
RUN sudo /usr/local/texlive/bin/x86_64-linux/tlmgr install hyperref
RUN sudo /usr/local/texlive/bin/x86_64-linux/tlmgr install makecell
RUN sudo /usr/local/texlive/bin/x86_64-linux/tlmgr install multirow
RUN sudo /usr/local/texlive/bin/x86_64-linux/tlmgr install scheme-basic
RUN sudo /usr/local/texlive/bin/x86_64-linux/tlmgr install siunitx
RUN sudo /usr/local/texlive/bin/x86_64-linux/tlmgr install type1cm
RUN sudo /usr/local/texlive/bin/x86_64-linux/tlmgr install todonotes
RUN sudo /usr/local/texlive/bin/x86_64-linux/tlmgr install enumitem
RUN sudo /usr/local/texlive/bin/x86_64-linux/tlmgr install lineno

FROM base as python_packages

ENV ASTROPATH_DEPENDENCY_COMMIT a23eaed9a927dfdba11a8c708d63959f587b1edd
ARG GITHUB_TOKEN

#add python packages
RUN sudo pip install --upgrade pip
RUN \
  sudo git config --global url."https://astropathjhujenkins:${GITHUB_TOKEN}@github".insteadOf https://github && \
  sudo pip install "astropath-calibration[test] @ git+https://github.com/AstropathJHU/microscopealignment@${ASTROPATH_DEPENDENCY_COMMIT}" && \
  sudo git config --global --unset url."https://astropathjhujenkins:${GITHUB_TOKEN}@github".insteadOf
RUN sudo pip uninstall -y astropath-calibration

from base as final

copy --from=python_packages /usr/local/lib/python3.6/site-packages/ /usr/local/lib/python3.6/site-packages/
copy --from=latex_packages /usr/local/texlive/ /usr/local/texlive/
copy --from=libvips /usr/local/bin/*vips* /usr/local/bin/
copy --from=libvips /usr/local/lib/*vips* /usr/local/lib/
ENV PATH=/usr/local/texlive/bin/x86_64-linux:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/lib/:${LD_LIBRARY_PATH}

CMD ["/bin/bash"]
