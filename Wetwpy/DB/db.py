import os
import mysql.connector
import json

# Ścieżka do folderu z obrazami
folder_path = "images/"

# Połączenie z bazą danych MySQL Workbench
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",  # Ustaw swoje hasło
    database="dogs_db"
)
cursor = conn.cursor()

# Tworzenie tabeli, jeśli nie istnieje
create_table_query = """
CREATE TABLE IF NOT EXISTS dogs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    breed TEXT,
    image MEDIUMBLOB,
    curiosity TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
)
"""
cursor.execute(create_table_query)

# Iteracja przez pliki w folderze
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        filename_without_extension = os.path.splitext(filename)[0]
        check_query = "SELECT id FROM dogs WHERE breed = %s"
        cursor.execute(check_query, (filename_without_extension,))
        existing_entry = cursor.fetchone()
        
        if not existing_entry:
            # Wczytaj dane z pliku JSON
            with open('curiosity.json', 'r') as file:
                data = json.load(file)
                
            # Iteracja przez każdy słownik w liście danych
            for item in data:
                if item['breed'] == filename_without_extension:
                    text = item['text']
                    # Wczytanie obrazu jako binarne dane
                    with open(os.path.join(folder_path, filename), 'rb') as image_file:
                        image_data = image_file.read()
                    insert_query = "INSERT INTO dogs (breed, image, curiosity) VALUES (%s, %s, %s)"
                    insert_values = (filename_without_extension, image_data, text)
                    cursor.execute(insert_query, insert_values)
                    print(f"Dodano do bazy danych: {filename_without_extension}")
        else:
            print(f"Zdjęcie {filename_without_extension} już istnieje w bazie danych.")

# Zatwierdzenie zmian w bazie danych
conn.commit()

# Zamknięcie połączenia
conn.close()
