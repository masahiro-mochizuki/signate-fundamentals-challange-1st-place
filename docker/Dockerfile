FROM continuumio/anaconda3:2019.03
COPY ./archive/requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y build-essential
COPY ./requirements_dev.txt .
RUN pip install -r requirements_dev.txt
RUN groupadd user && \
    useradd  user -g user -G sudo -m
USER user
RUN mkdir /home/user/work && chmod -R o+w /home/user/
WORKDIR /home/user/work
