
# Build image:    docker build -t fastapi-reqgen:latest .
# Run container:  docker run -p 8080:8000 -v .:/app/ fastapi-reqgen:latest

# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the local directory's contents into the container at /app
COPY . /app

RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run uvicorn when the container launches
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

  