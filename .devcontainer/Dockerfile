FROM mcr.microsoft.com/devcontainers/python:0-3.11-bullseye

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# Install PyTorch.
RUN pip install torch torchvision torchaudio

# Install spacy.
RUN pip install -U pip setuptools wheel
RUN pip install -U spacy
RUN python -m spacy download en_core_web_sm

# Copy the requirement.txt into the cwd and run pip to install all necessary packages.
COPY alternative_wordings/requirements.txt /tmp/conda-tmp/
RUN pip install --no-cache-dir --upgrade -r /tmp/conda-tmp/requirements.txt

COPY alternative_wordings/package.json /tmp/conda-tmp/
COPY alternative_wordings/package-lock.json /tmp/conda-tmp/
# RUN pip install npm
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
     && apt-get -y install --no-install-recommends npm
RUN npm install