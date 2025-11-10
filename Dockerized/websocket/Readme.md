# Prosty Serwer WebSocket

Prosty serwer WebSocket w Pythonie z użyciem `asyncio` i biblioteki `websockets`.

## 1. Instalacja

Przed uruchomieniem, zainstaluj wszystkie wymagane zależności:

```bash
    pip install -r requirements.txt
```

## 2. Uruchomienie

Aplikacja składa się z serwera, (opcjonalnie) klienta testowego oraz testów.

### Uruchomienie serwera

W pierwszym oknie terminala uruchom główny serwer:

```bash
    python server.py
```

Serwer będzie nasłuchiwał na `ws://localhost:8765`.

### Uruchomienie klienta testowego

(Wymaga działającego serwera) W drugim oknie terminala możesz uruchomić przykładowego klienta, aby połączyć się z serwerem i wysłać komendy:

```bash
    python client.py
```

### Uruchomienie testów

Aby zweryfikować poprawność działania logiki oraz serwera, uruchom testy za pomocą `pytest`:

```bash
    pytest -v
```