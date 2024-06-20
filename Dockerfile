# 
FROM python:3.9

# 
WORKDIR /app

# 
COPY ./requirements.txt /app/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# 
COPY ./app /app/app

COPY ./serviceAccountFireStore.json /app/serviceAccountFireStore.json

# 
CMD ["fastapi", "run", "app/app.py", "--port", "8080"]