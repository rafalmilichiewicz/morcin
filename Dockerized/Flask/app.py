from flask import Flask, jsonify, request
import mysql.connector

app = Flask(__name__)

def get_db_connection():
    return mysql.connector.connect(
        host="db",
        port=3306,
        user="root",
        password="password",
        database="dogs_db"
    )

@app.route('/dogs', methods=['GET'])
def get_dogs():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM dogs")
    results = cursor.fetchall()
    dogs = [{'id': result[0], 'breed': result[1], 'obrazek': result[2].hex(), 'curiosity': result[3]} for result in results]
    conn.close()
    return jsonify(dogs)

@app.route('/dogs', methods=['POST'])
def add_dog():
    data = request.json
    conn = get_db_connection()
    cursor = conn.cursor()
    insert_query = "INSERT INTO dogs (breed, obrazek, curiosity) VALUES (%s, %s, %s)"
    cursor.execute(insert_query, (data['breed'], bytes.fromhex(data['obrazek']), data['curiosity']))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Dog added successfully'}), 200

@app.route('/dogs/<breed>', methods=['DELETE'])
def delete_dog(breed):
    conn = get_db_connection()
    cursor = conn.cursor()
    delete_query = "DELETE FROM dogs WHERE breed = %s"
    cursor.execute(delete_query, (breed,))
    conn.commit()
    conn.close()
    return jsonify({'message': f'Dog {breed} deleted successfully'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
