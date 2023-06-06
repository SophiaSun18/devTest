FROM python:3.10

# Set up the working directory.
WORKDIR /code
COPY ./ /code

# RUN npm install
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
     && apt-get -y install --no-install-recommends npm
RUN npm install

# Add the mBART50 model and extract the inner file
ADD https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.ft.nn.tar.gz /alternative_wordings/
RUN tar -xJf /alternative_wordings/mbart50.ft.nn.tar.gz -C /alternative_wordings
RUN make -C /alternative_wordings all

# RUN mkdir -p /alternative_wordings \
#     && curl -SL https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.ft.nn.tar.gz \
#     | tar -xJC /alternative_wordings \
#     && make -C /alternative_wordings all

# Update pip
RUN pip install --upgrade pip
# Install packages
RUN pip install torch torchvision torchaudio
RUN pip install -U pip setuptools wheel
RUN pip install -U spacy
RUN python -m spacy download en_core_web_sm

# Copy the requirement.txt into the WORKDIR and run pip to install all necessary packages.
RUN pip install --no-cache-dir --upgrade -r /code/alternative_wordings/requirements.txt