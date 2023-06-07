# Using python 3.11 will cause a ValueError on fairseq class.
FROM python:3.10

# Set up the working directory.
WORKDIR /code
COPY ./ /code

# RUN npm install
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
     && apt-get -y install --no-install-recommends npm
RUN npm install

# Install packages
RUN pip install -U pip setuptools wheel spacy
RUN pip install torch torchvision torchaudio tensorboardX
RUN python -m spacy download en_core_web_sm

# Copy the requirement.txt into the WORKDIR and run pip to install all necessary packages.
RUN pip install --no-cache-dir --upgrade -r /code/alternative_wordings/requirements.txt

# Add the mBART50 model
# RUN wget -P /alternative_wordings/ https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.ft.nn.tar.gz 
# RUN tar -C /alternative_wordings -xz /alternative_wordings/mbart50.ft.nn.tar.gz
