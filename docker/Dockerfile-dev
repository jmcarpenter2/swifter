FROM python:3.9
ADD requirements-dev.txt /build/requirements.txt
WORKDIR /build/
RUN pip install -r requirements.txt
WORKDIR /mnt/
ENV PYTHONPATH "${PYTHONPATH}:/mnt"
