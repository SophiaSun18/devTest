FROM python:3.11

WORKDIR /code
COPY ./ /code

RUN pip install --no-cache-dir --upgrade -r /requirements.txt
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]