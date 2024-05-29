-- Informacja o wywołaniu skryptu
SELECT 'Inicjalizacja bazy danych rozpoczęta...' AS 'Komunikat';

-- Tworzenie bazy danych, tabeli, itp.
CREATE DATABASE IF NOT EXISTS dogs_db;

USE dogs_db;

CREATE TABLE IF NOT EXISTS dogs(
    id INT AUTO_INCREMENT PRIMARY KEY,
    breed VARCHAR(100) NOT NULL,
    curiosity VARCHAR(100) NOT NULL,
    obrazek BLOB(65535) NOT NULL
);

-- Informacja o zakończeniu wywołania skryptu
SELECT 'Inicjalizacja bazy danych zakończona.' AS 'Komunikat';
