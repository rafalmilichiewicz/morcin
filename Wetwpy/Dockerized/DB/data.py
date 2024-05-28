import os
import json
import requests
import mysql.connector

# Ścieżka do folderu z obrazami
folder_path = "images/"

# Pobieranie zmiennych środowiskowych dla konfiguracji bazy danych
db_host = os.getenv('DB_HOST', 'db')
db_port = os.getenv('DB_PORT', '3306')  # Adjust the port if needed
add_dog_url = f"http://{db_host}:{db_port}/dogs"

# Establishing connection to MySQL database
conn = mysql.connector.connect(
    host=db_host,
    port=db_port,
    user='root',  # Adjust if needed
    password='password',  # Adjust if needed
    database='dogs_db'  # Adjust if needed
)
cursor = conn.cursor()

try:
    # Iteracja przez pliki w folderze
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            filename_without_extension = os.path.splitext(filename)[0]
            # Wczytaj dane z pliku JSON
            with open('curiosity.json', 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Iteracja przez każdy słownik w liście danych
            for item in data:
                if item['breed'] == filename_without_extension:
                    text = item['text']
                    # Wczytanie obrazu jako binarne dane
                    with open(os.path.join(folder_path, filename), 'rb') as image_file:
                        image_data = image_file.read()

                    # Tworzenie JSON payloadu
                    payload = {
                        'breed': filename_without_extension,
                        'image': image_data.hex(),
                        'curiosity': text
                    }

                    # Wysyłanie POST requestu
                    response = requests.post(add_dog_url, json=payload)
                    if response.status_code == 200:
                        print(f"Dodano do bazy danych: {filename_without_extension}")
                    else:
                        print(f"Błąd podczas dodawania do bazy danych: {response.text}")
except Exception as e:
    print(f"Error: {e}")
finally:
    # Closing the connection
    conn.close()
