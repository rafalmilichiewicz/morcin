-- Utwórz bazę danych, jeżeli nie istnieje
CREATE DATABASE IF NOT EXISTS dogs_db;

-- Użyj bazy danych
USE dogs_db;

-- Utwórz tabelę dogs, jeżeli nie istnieje
CREATE TABLE IF NOT EXISTS dogs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    breed VARCHAR(100) NOT NULL,
    image LONGBLOB NOT NULL,
    curiosity TEXT NOT NULL
);
