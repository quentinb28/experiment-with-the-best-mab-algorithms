FROM python:3.7
ADD . /app
WORKDIR /app
RUN pip3 install --no-cache-dir -r requirements.txt
EXPOSE 8081
CMD ["python", "app.py"]
