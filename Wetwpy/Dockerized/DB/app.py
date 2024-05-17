from flask import Flask, jsonify, request
import mysql.connector

app = Flask(__name__)

@app.route('/dogs', methods=['GET'])
def get_dogs():
    conn = mysql.connector.connect(
        host="db",
        user="root",
        password="password",  # Ustaw swoje hasło
        database="dogs_db"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM dogs")
    results = cursor.fetchall()
    dogs = [{'id': result[0], 'breed': result[1], 'image': result[2].hex(), 'curiosity': result[3]} for result in results]
    conn.close()
    return jsonify(dogs)

@app.route('/dog/<breed>', methods=['GET'])
def get_dog(breed):
    conn = mysql.connector.connect(
        host="db",
        user="root",
        password="password",  # Ustaw swoje hasło
        database="dogs_db"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM dogs WHERE breed = %s", (breed,))
    result = cursor.fetchone()
    if result:
        dog = {'id': result[0], 'breed': result[1], 'image': result[2].hex(), 'curiosity': result[3]}
    else:
        dog = None
    conn.close()
    return jsonify(dog)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
