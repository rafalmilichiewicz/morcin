FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir Flask mysql-connector-python requests

# Ensure the script has executable permissions
RUN chmod +x /app/init_db.py

# Run the initialization script and then start the Flask server
CMD ["sh", "-c", "python init_db.py && python app.py "]

EXPOSE 5000
