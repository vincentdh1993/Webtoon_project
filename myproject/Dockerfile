# pull official base image
FROM python:3.8


# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1



COPY . /usr/src/app/
# install dependencies
RUN pip install --upgrade pip
RUN python --version
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt --timeout 10000
# RUN pip install --default-timeout=10000 torch