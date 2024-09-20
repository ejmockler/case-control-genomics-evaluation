# Use the official Hail Docker image as a base
FROM hailgenetics/hail:latest

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run main.py when the container launches
CMD ["python", "main.py"]