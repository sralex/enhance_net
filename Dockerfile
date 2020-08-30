FROM continuumio/miniconda3

ENV APP_ROOT /app/
ENV CONFIG_ROOT /config/

RUN mkdir ${CONFIG_ROOT}
RUN mkdir ${APP_ROOT}

ADD app/ ${APP_ROOT}

ADD /app/environment.yml ${CONFIG_ROOT}/environment.yml


RUN /opt/conda/bin/conda env create -f ${CONFIG_ROOT}/environment.yml

WORKDIR ${APP_ROOT}

ENTRYPOINT ["conda", "run", "-n", "keras_gunicorn_nginx_flask", "gunicorn", "--bind=0.0.0.0:9000","main:app"]
