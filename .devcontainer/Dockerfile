FROM python:3.10

# Set up the working directory.
WORKDIR /code
COPY ./ /code

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# RUN npm install
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
     && apt-get -y install --no-install-recommends npm
RUN npm install

# Update pip
RUN pip install --upgrade pip
# Install packages
RUN pip install torch torchvision torchaudio
RUN pip install -U pip setuptools wheel
RUN pip install -U spacy
RUN python -m spacy download en_core_web_sm

# Copy the requirement.txt into the WORKDIR and run pip to install all necessary packages.
RUN pip install --no-cache-dir --upgrade -r /code/alternative_wordings/requirements.txt