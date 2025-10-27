import os
import json
import mysql.connector
from .env import get_db_connection
def example_dogs():
    # Path to the folder with images
    folder_path = "db/images/"

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Directory '{folder_path}' does not exist.")
        return

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Iterating through files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                filename_without_extension = os.path.splitext(filename)[0]
                
                # Read data from JSON file
                with open('db/curiosity.json', 'r', encoding='utf-8') as file:
                    data = json.load(file)

                # Iterate through each dictionary in the data list
                for item in data:
                    if item['breed'] == filename_without_extension:
                        text = item['text']
                        
                        # Read the image as binary data
                        with open(os.path.join(folder_path, filename), 'rb') as image_file:
                            image_data = image_file.read()

                        # Insert data into the database
                        insert_query = "INSERT INTO dogs (breed, image, curiosity) VALUES (%s, %s, %s)"
                        cursor.execute(insert_query, (filename_without_extension, image_data, text))
                        conn.commit()
                        print(f"Added to the database: {filename_without_extension}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Closing the connection
        conn.close()