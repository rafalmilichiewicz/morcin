import asyncio
import json
from datetime import datetime, timezone
from dataclasses import dataclass, asdict #Generuje samemu wszystkie konstruktory

# Używamy websockets.server.Protocol, aby uzyskać dostęp do adnotacji typów
# dla obiektu 'websocket' w handlerze
from websockets.server import serve, Protocol

@dataclass
class ServerInfo:
    """
    Prosta klasa danych (dataclass) przechowująca informacje o serwerze.
    Używa adnotacji typów.
    """
    status: str
    server_time: str
    version: str

class InfoProvider:
    """
    Klasa odpowiedzialna za *logikę* generowania informacji o serwerze.
    Oddzielenie jej od serwera WebSocket ułatwia testowanie.
    """
    def __init__(self, version: str = "1.0.0"):
        # Prywatne pole z adnotacją typu
        self._version: str = version

    def get_server_info(self) -> ServerInfo:
        """
        Generuje i zwraca aktualne informacje o serwerze.
        Zwraca obiekt ServerInfo, co jest jasno określone przez adnotację.
        """
        return ServerInfo(
            status="ok",
            server_time=datetime.now(timezone.utc).isoformat(),
            version=self._version
        )

class WebSocketServerApp:
    """
    Główna klasa aplikacji serwera, enkapsulująca logikę połączeń.
    """
    def __init__(self, host: str, port: int, info_provider: InfoProvider):
        self.host: str = host
        self.port: int = port
        self.info_provider: InfoProvider = info_provider
        print(f"Inicjalizacja serwera na {host}:{port}...")

    async def _handler(self, websocket: Protocol) -> None:
        """
        Asynchroniczny handler dla każdego podłączonego klienta.
        'websocket: Protocol' to adnotacja typu dla przychodzącego połączenia.
        Zwraca 'None', jak wskazuje adnotacja '-> None'.
        """
        print(f"Klient połączony: {websocket.remote_address}")
        
        try:
            # Pętla nasłuchująca na wiadomości od klienta
            async for message in websocket:
                print(f"Otrzymano wiadomość: {message}")
                
                # Prosty routing poleceń
                if str(message) == "GET_INFO":
                    # Pobieramy dane z naszej klasy logicznej
                    info_object: ServerInfo = self.info_provider.get_server_info()
                    
                    # Konwertujemy dataclass na słownik i potem na JSON
                    response: str = json.dumps(asdict(info_object))
                    await websocket.send(response)
                
                elif str(message) == "PING":
                    await websocket.send(json.dumps({"response": "PONG"}))
                
                else:
                    error_msg: dict[str, str] = {
                        "error": "Unknown command",
                        "received": str(message)
                    }
                    await websocket.send(json.dumps(error_msg))

        except Exception as e:
            print(f"Błąd połączenia: {e}")
        finally:
            print(f"Klient rozłączony: {websocket.remote_address}")

    async def start(self) -> None:
        """
        Uruchamia główną pętlę serwera.
        """
        print(f"Serwer nasłuchuje na ws://{self.host}:{self.port}")
        # 'serve' to asynchroniczny menedżer kontekstu
        async with serve(self._handler, self.host, self.port):
            # Utrzymuje serwer przy życiu na zawsze
            await asyncio.Future()

# Punkt uruchomienia skryptu
if __name__ == "__main__":
    # 1. Utwórz dostawcę informacji (logika)
    provider = InfoProvider(version="1.0.1-beta")
    
    # 2. Utwórz aplikację serwera (transport)
    app = WebSocketServerApp("0.0.0.0", 8765, provider)
    
    # 3. Uruchom pętlę zdarzeń asyncio
    try:
        asyncio.run(app.start())
    except KeyboardInterrupt:
        print("\nZamykanie serwera...")