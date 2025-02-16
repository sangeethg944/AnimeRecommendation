# Use a Python base image
FROM python:3.9

# Install R and Shiny
#RUN apt-get update && apt-get install -y r-base r-cran-shiny && apt-get clean

# Set the working directory
WORKDIR /app

# Copy the application files
COPY requirements.txt .
COPY df_animes_updated.csv .
COPY app.py .
COPY kmeans_model.pkl .
COPY scaler.pkl .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8000

# Run the Shiny app
CMD ["shiny", "run", "/app/app.py", "--host=0.0.0.0", "--port=8000"]
#CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & shiny run .\app.py --server.port=8501 --server.enableCORS=false"]

#CMD ["shiny", "run", "/app/app.py", "--host=127.0.0.1", "--port=8000"]