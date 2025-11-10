import asyncio
import websockets
import json

async def a_talk_to_server() -> None:
    uri = "ws://127.0.0.1:8765"
    try:
        async with websockets.connect(uri) as websocket:
            print("Połączono z serwerem.")

            # 1. Zapytaj o informacje
            await websocket.send("GET_INFO")
            response = await websocket.recv()
            print(f"Otrzymano info: {json.loads(response)}")

            # 2. Wyślij komendę PING
            await websocket.send("PING")
            response = await websocket.recv()
            print(f"Otrzymano ping: {json.loads(response)}")

            # 3. Wyślij błędną komendę
            await websocket.send("COKOLWIEK")
            response = await websocket.recv()
            print(f"Otrzymano błąd: {json.loads(response)}")

    except ConnectionRefusedError:
        print("Nie można połączyć się z serwerem. Czy jest uruchomiony?")

if __name__ == "__main__":
    asyncio.run(a_talk_to_server())