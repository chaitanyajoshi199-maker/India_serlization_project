from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class DatalogicImageSource:
    """Live Matrix 220 image source.

    Matrix 220 sends image packets over TCP. Each complete BMP payload is:
        [optional protocol/header bytes] [BMP image bytes] [optional terminator]

    Every received BMP is first saved to ``image_directory``. Only after the
    file is successfully written is it loaded with OpenCV and returned as the
    frame for inspection.
    """

    def __init__(
        self,
        matrix_ip: str,
        image_port: int = 51237,
        image_format: str = "bmp",
        header_hex: str = "02",
        terminator_hex: str = "0D0A",
        reconnect_delay_seconds: float = 1.0,
        socket_timeout_seconds: float = 2.0,
        image_directory: str | Path = "datalogic_images",
        save_to_disk: bool = False,
    ) -> None:
        self.matrix_ip = str(matrix_ip)
        self.image_port = int(image_port)
        self.image_format = str(image_format).lower()
        self.header_hex = header_hex
        self.terminator_hex = terminator_hex
        self.reconnect_delay_seconds = float(reconnect_delay_seconds)
        self.socket_timeout_seconds = float(socket_timeout_seconds)
        self.save_to_disk = save_to_disk

        self.image_directory = Path(image_directory)
        if self.save_to_disk:
            self.image_directory.mkdir(parents=True, exist_ok=True)

        self._sock: Optional[socket.socket] = None
        self._buffer = bytearray()
        self._opened = False
        self._frame_counter = 0

    def open(self) -> bool:
        """Open the TCP connection to the Matrix image channel."""
        self.release()

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.socket_timeout_seconds)
            sock.connect((self.matrix_ip, self.image_port))

            self._sock = sock
            self._opened = True
            self._buffer.clear()

            print(
                f"[DATALOGIC] Connected to Matrix image channel "
                f"{self.matrix_ip}:{self.image_port}"
            )
            if self.save_to_disk:
                print(
                    f"[DATALOGIC] Raw images saved to: "
                    f"{self.image_directory.resolve()}"
                )
            else:
                print("[DATALOGIC] Direct in-memory image decoding active (zero disk writes).")
            return True

        except (OSError, socket.timeout) as exc:
            self._opened = False
            self._sock = None
            print(
                f"[DATALOGIC] Image TCP connection failed "
                f"{self.matrix_ip}:{self.image_port} -> {exc}"
            )
            return False

    def isOpened(self) -> bool:
        return self._opened and self._sock is not None

    def _recv_more(self) -> bool:
        """Receive another TCP chunk into the internal packet buffer."""
        if not self.isOpened():
            return False

        try:
            chunk = self._sock.recv(65536)
        except socket.timeout:
            return False
        except OSError as exc:
            print(f"[DATALOGIC] TCP receive error -> {exc}")
            self._opened = False
            return False

        if not chunk:
            print("[DATALOGIC] Matrix image connection closed by peer.")
            self._opened = False
            return False

        self._buffer.extend(chunk)
        return True

    def _extract_bmp(self) -> Optional[bytes]:
        """Extract one complete BMP image from the TCP receive buffer."""
        if len(self._buffer) < 6:
            return None

        # Find the BMP signature. This also skips an optional Matrix packet
        # header that precedes the image buffer.
        bmp_start = self._buffer.find(b"BM")
        if bmp_start < 0:
            # Keep the last byte in case it is the beginning of "BM" after the
            # next recv().
            if self._buffer:
                self._buffer[:] = self._buffer[-1:]
            return None

        if len(self._buffer) < bmp_start + 6:
            return None

        image_size = int.from_bytes(
            self._buffer[bmp_start + 2 : bmp_start + 6],
            byteorder="little",
            signed=False,
        )

        # Sanity protection against malformed packet lengths.
        if image_size < 54 or image_size > 100 * 1024 * 1024:
            del self._buffer[: bmp_start + 2]
            return None

        packet_end = bmp_start + image_size
        if len(self._buffer) < packet_end:
            return None

        image_bytes = bytes(self._buffer[bmp_start:packet_end])

        # Remove image bytes from the stream. Any following terminator or the
        # beginning of the next packet remains buffered for the next read.
        del self._buffer[:packet_end]

        return image_bytes

    def _save_image(self, image_bytes: bytes) -> Optional[Path]:
        """Persist one raw BMP and return its exact path."""
        self._frame_counter += 1
        filename = f"frame_{self._frame_counter:06d}.bmp"
        path = self.image_directory / filename

        try:
            path.write_bytes(image_bytes)
        except OSError as exc:
            print(f"[DATALOGIC] Failed to save image {path} -> {exc}")
            return None

        return path

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Receive, save, then load one Matrix image.

        Returns:
            (True, ndarray) when a complete image was received and successfully
            reloaded from disk; otherwise (False, None).
        """
        if not self.isOpened():
            if not self.open():
                time.sleep(self.reconnect_delay_seconds)
                return False, None

        while self.isOpened():
            image_bytes = self._extract_bmp()

            if image_bytes is not None:
                if self.save_to_disk:
                    self._save_image(image_bytes)

                # High-speed in-memory decode (avoids disk filling and I/O latency)
                frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    print("[DATALOGIC] Image bytes could not be decoded.")
                    return False, None

                return True, frame

            if not self._recv_more():
                if not self.isOpened():
                    time.sleep(self.reconnect_delay_seconds)
                return False, None

        return False, None

    def release(self) -> None:
        """Close the Matrix image connection."""
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass

        self._sock = None
        self._opened = False
        self._buffer.clear()
