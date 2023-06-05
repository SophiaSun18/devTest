FROM python:3.10

# Set up the working directory.
# WORKDIR /app

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

## Pip dependencies
# Update pip
RUN pip install --upgrade pip
# Install packages
RUN pip install torch torchvision torchaudio
RUN pip install -U pip setuptools wheel
RUN pip install -U spacy
RUN python -m spacy download en_core_web_sm

COPY ./alternative_wordings/requirements.txt .
RUN pip install -r requirements.txt

# COPY alternative_wordings/package.json /tmp/conda-tmp/
# COPY alternative_wordings/package-lock.json /tmp/conda-tmp/
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
     && apt-get -y install --no-install-recommends npm
RUN npm install
CMD [ "npm","start" ]