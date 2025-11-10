import pytest  # Poprawka 1: Poprawny import
import pytest_asyncio
import asyncio
import json
import websockets # Potrzebny do klienta testowego
from datetime import datetime, timezone  # Poprawka 2: Dodano import timezone
from typing import AsyncGenerator      # Poprawka 3: Import dla adnotacji typu

# Importujemy klasy z naszego modułu serwera
from server import InfoProvider, ServerInfo, WebSocketServerApp

# -- Test 1: Testowanie logiki (klasa synchroniczna) --

def test_info_provider_version() -> None:
    """ Testuje, czy InfoProvider poprawnie ustawia wersję. """
    provider = InfoProvider(version="test-v2")
    info: ServerInfo = provider.get_server_info()
    
    assert info.version == "test-v2"
    assert info.status == "ok"

def test_info_provider_time() -> None:
    """ Testuje, czy InfoProvider generuje poprawny czas. """
    provider = InfoProvider()
    info: ServerInfo = provider.get_server_info()
    
    # Sprawdzamy, czy czas jest w formacie ISO i jest aktualny
    time_from_info = datetime.fromisoformat(info.server_time)
    # Używamy timezone.utc (teraz zaimportowane)
    time_diff_seconds = (datetime.now(timezone.utc) - time_from_info).total_seconds()
    
    assert abs(time_diff_seconds) < 1 # Czas nie powinien różnić się o więcej niż 1s


# -- Test 2: Testowanie serwera (test integracyjny) --

@pytest_asyncio.fixture
# Poprawka 3: Poprawna adnotacja typu dla asynchronicznego generatora
async def running_server() -> AsyncGenerator[str, None]:
    """
    Fixtura Pytest, która uruchamia nasz serwer w tle na czas trwania testu.
    Używa portu 8766 (innego niż główny), aby uniknąć konfliktów.
    """
    test_port = 8766
    provider = InfoProvider(version="test-server-v1")
    app = WebSocketServerApp("localhost", test_port, provider)
    
    # Uruchom serwer jako zadanie w tle
    server_task = asyncio.create_task(app.start())
    
    # Daj serwerowi chwilę na uruchomienie
    await asyncio.sleep(0.1)
    
    # Zwróć adres URI do testów
    yield f"ws://localhost:{test_port}"
    
    # Sprzątanie po teście: anuluj zadanie serwera
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        print("Serwer testowy zatrzymany.")


@pytest.mark.asyncio  # Poprawka 1: Używamy 'pytest', a nie 'test_server'
async def test_server_get_info(running_server: str) -> None:
    """
    Testuje komendę 'GET_INFO'.
    Wstrzykuje fixturę 'running_server'.
    """
    uri = running_server
    async with websockets.connect(uri) as ws:
        await ws.send("GET_INFO")
        response_str: str = await ws.recv()
        
        data: dict = json.loads(response_str)
        
        assert data["status"] == "ok"
        assert data["version"] == "test-server-v1"
        assert "server_time" in data

@pytest.mark.asyncio  # Poprawka 1: Używamy 'pytest', a nie 'test_server'
async def test_server_unknown_command(running_server: str) -> None:
    """ Testuje wysłanie nieznanej komendy. """
    uri = running_server
    async with websockets.connect(uri) as ws:
        await ws.send("BŁĘDNA_KOMENDA")
        response_str: str = await ws.recv()
        
        data: dict = json.loads(response_str)
        
        assert data["error"] == "Unknown command"
        assert data["received"] == "BŁĘDNA_KOMENDA"