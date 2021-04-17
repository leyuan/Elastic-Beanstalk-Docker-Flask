# FROM amazon/aws-eb-python:3.4.2-onbuild-3.5.1
# EXPOSE 8080

FROM python:3.6
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]
EXPOSE 8080