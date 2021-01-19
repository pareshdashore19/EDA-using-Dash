FROM python:3.8
COPY requirements.txt /tmp/
COPY  . /app
WORKDIR "/app"
EXPOSE 8050
RUN pip install -r /tmp/requirements.txt
ENTRYPOINT [ "python3" ]
CMD [ "index.py" ]

