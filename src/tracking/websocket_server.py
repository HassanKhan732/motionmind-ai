"""
MotionMind AI - Transport Module: WebSocket Broadcaster
Runs a lightweight, threaded WebSocket server to stream real-time hand transform telemetry to Three.js.
"""

import asyncio
import json
import threading
import websockets


class HandStreamServer:
    """
    Thread-safe WebSocket server that broadcasts hand coordinates to all connected web clients.
    """

    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        self.thread = None
        self.is_running = False

    async def _handler(self, websocket):
        """Register newly connected browser client and wait until disconnection."""
        self.clients.add(websocket)
        print(f"[WebSocket] Client connected: {websocket.remote_address} (Total: {len(self.clients)})")
        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)
            print(f"[WebSocket] Client disconnected. (Remaining: {len(self.clients)})")

    async def _send_to_all(self, message):
        """Send message concurrently to all active clients."""
        if not self.clients:
            return

        dead_clients = set()
        for client in self.clients:
            try:
                await client.send(message)
            except Exception:
                dead_clients.add(client)

        if dead_clients:
            self.clients.difference_update(dead_clients)

    def broadcast(self, data_dict):
        """
        Thread-safe method called from OpenCV main thread to push data packets.
        """
        if not self.is_running or not self.loop or not self.clients:
            return

        try:
            message = json.dumps(data_dict)
            asyncio.run_coroutine_threadsafe(self._send_to_all(message), self.loop)
        except Exception:
            pass

    def _start_loop(self):
        """Server execution loop running on a dedicated daemon thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        async def run_server():
            async with websockets.serve(self._handler, self.host, self.port):
                print(f"[WebSocket] Server listening on ws://{self.host}:{self.port}")
                await asyncio.Future()  # Keep running indefinitely

        try:
            self.loop.run_until_complete(run_server())
        except Exception:
            pass

    def start(self):
        """Start the WebSocket server in a background thread."""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop server and clean up loop."""
        self.is_running = False
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
