FROM python:3.6-slim
LABEL maintainer="heidonomm@gmail.com"

WORKDIR /qait/

# Install build utilities
RUN apt-get update  && apt-get -y install build-essential locales curl git-all

# Set Language settings for Inform (TextWorld engine)
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN export LC_ALL=C

COPY requirements.txt .

# Install Dependencies
RUN pip install torch==1.5.1+cu92 torchvision==0.6.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en
RUN pip install https://github.com/Microsoft/TextWorld/archive/rebased-interactive-qa.zip

RUN export LC_ALL="en_US.UTF-8" && \
    export LC_CTYPE="en_US.UTF-8"

COPY . .
RUN ls
RUN pwd

RUN python --version
RUN pip --version
RUN pip show textworld
RUN pip show gym

# Training
CMD ["python", "train.py", "./"]
# FROM continuumio/miniconda3:latest
# LABEL maintainer="heidonomm@gmail.com"

# WORKDIR /qait/

# # Install build utilities
# RUN apt-get update  && apt-get -y install build-essential locales
# RUN locale-gen en_US.UTF-8
# ENV LANG en_US.UTF-8
# ENV LANGUAGE en_US:en
# ENV LC_ALL en_US.UTF-8
# RUN export LC_ALL=C


# COPY environment.yml ./

# # RUN pip install https://github.com/Microsoft/TextWorld/archive/rebased-interactive-qa.zip
# RUN conda env create -f environment.yml
# RUN echo "conda activate qait" > ~/.bashrc

# ## download evaluation set set
# RUN wget https://aka.ms/qait-testset && unzip qait-testset

# ENV PATH /opt/conda/envs/qait/bin:$PATH

# COPY . .

# RUN chmod +x train.py

# # Training
# CMD ["python", "train.py ./"]


