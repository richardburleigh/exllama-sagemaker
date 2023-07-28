FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as build
ARG APPLICATION_STATE_PATH="/data"
ENV APPLICATION_STATE_PATH=$APPLICATION_STATE_PATH \
    CONTAINER_MODEL_PATH=$APPLICATION_STATE_PATH/model \
    CONTAINER_SESSIONS_PATH=$APPLICATION_STATE_PATH/exllama_sessions

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y nginx ca-certificates wget ninja-build python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Create application state directories
RUN mkdir -p $APPLICATION_STATE_PATH \
    && mkdir -p $CONTAINER_MODEL_PATH \
    && mkdir -p $CONTAINER_SESSIONS_PATH

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

COPY . /app

WORKDIR /app

# Install python packages
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip install -r requirements-web.txt

STOPSIGNAL SIGINT
ENTRYPOINT ["python3", "serve"]
