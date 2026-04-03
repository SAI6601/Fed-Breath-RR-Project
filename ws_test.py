import asyncio
import websockets
import json

async def test_ws():
    async with websockets.connect("ws://127.0.0.1:8000/ws") as ws:
        for i in range(3):
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"Data keys: {data.keys()}")
            print(f"wave is none? {data.get('wave') is None}")

asyncio.get_event_loop().run_until_complete(test_ws())
