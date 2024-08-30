# The MIT License (MIT)
#
# Copyright (c) 2024-2025 Kris Jusiak <kris@jusiak.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ```
# docker build -t perf .
# ```

# ```
# docker run \
#   --rm \
#   --user $(id -u):$(id -g) \
#   --privileged \
#   --network=host \
#   -e DISPLAY=${DISPLAY} \
#   -v ${PWD}:${PWD} \
#   -w ${PWD} \
#   perf
# ```

FROM ubuntu:24.04
ARG CLANG=20
ARG GCC=14

RUN apt-get update && apt-get install -y \
  gnupg software-properties-common lsb-release ca-certificates \
  python3-pip python3-venv cmake \
  wget bzip2 vim pkg-config dbus-x11 git gh \
  kmod linux-headers-generic linux-tools-generic linux-tools-*-generic util-linux hwloc msr-tools \
  llvm-dev libpfm4 libpapi-dev papi-tools libipt-dev \
  libgoogle-perftools-dev google-perftools valgrind kcachegrind coz-profiler pahole likwid hotspot hyperfine \
  graphviz gnuplot coreutils imagemagick libsixel-bin

RUN python3 -m venv /venv
RUN /venv/bin/pip install \
  pyperf osaca \
  jupyter numpy pandas matplotlib ansi2html

RUN wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
    gpg --dearmor -o /usr/share/keyrings/intel-oneapi-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
    tee /etc/apt/sources.list.d/oneAPI.list
RUN apt-get update && apt-get install -y libnss3 intel-oneapi-vtune

RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh $CLANG && rm llvm.sh
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && apt update && apt install -y g++-$GCC

ENV PATH="/venv/bin:$PATH"
ENV DISPLAY=:0
