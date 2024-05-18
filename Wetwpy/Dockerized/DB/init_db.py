import os
import mysql.connector
import json

# Ścieżka do folderu z obrazami
folder_path = "images/"

# Pobieranie zmiennych środowiskowych dla konfiguracji bazy danych
db_host = os.getenv('DB_HOST', 'db')
db_user = os.getenv('DB_USER', 'root')
db_password = os.getenv('DB_PASSWORD', 'password')
db_name = os.getenv('DB_NAME', 'dogs_db')

try:
    # Połączenie z bazą danych
    conn = mysql.connector.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name
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
        if filename.endswith((".jpg", ".png", ".jpeg")):
            filename_without_extension = os.path.splitext(filename)[0]
            check_query = "SELECT id FROM dogs WHERE breed = %s"
            cursor.execute(check_query, (filename_without_extension,))
            existing_entry = cursor.fetchone()

            if not existing_entry:
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
                        insert_query = "INSERT INTO dogs (breed, image, curiosity) VALUES (%s, %s, %s)"
                        insert_values = (filename_without_extension, image_data, text)
                        cursor.execute(insert_query, insert_values)
                        print(f"Dodano do bazy danych: {filename_without_extension}")
            else:
                print(f"Zdjęcie {filename_without_extension} już istnieje w bazie danych.")

    # Zatwierdzenie zmian w bazie danych
    conn.commit()

except mysql.connector.Error as err:
    print(f"MySQL error: {err}")
except Exception as e:
    print(f"Error: {e}")
finally:
    try:
        if conn and conn.is_connected():
            # Zamknięcie połączenia
            cursor.close()
            conn.close()
    except NameError:
        pass
