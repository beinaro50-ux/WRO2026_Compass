#!/usr/bin/env python3
import serial, threading, time, sys, tty, termios

SERIAL_PORT = "/dev/esp32"
BAUD_RATE = 115200
STEER_RANGE_DEG = 135 # adjust to your steering linkage's real limit

class RobotController:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2)
        self.encoder_ticks = 0
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self):
        while self._running:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
            except Exception:
                continue
            if line.startswith("ENC,"):
                try:
                    with self._lock:
                        self.encoder_ticks = int(line.split(",")[1])
                except (IndexError, ValueError):
                    pass

    def get_encoder(self):
        with self._lock:
            return self.encoder_ticks

    def set_linear(self, linear):
        linear = max(-1.0, min(1.0, linear))
        self.ser.write(f"M,{int(linear * 255)}\n".encode())

    def set_angular(self, angular):
        angular = max(-1.0, min(1.0, angular))
        angle = max(0, min(180, 90 + int(angular * STEER_RANGE_DEG)))
        self.ser.write(f"S,{angle}\n".encode())

    def close(self):
        self._running = False
        self.set_linear(0.0)
        self.set_angular(0.0)
        time.sleep(0.1)
        self.ser.close()

def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def main():
    robot = RobotController(SERIAL_PORT, BAUD_RATE)
    linear, angular = 0.0, 0.0
    print("W/S linear, A/D angular, SPACE stop, C center, Q quit")
    try:
        while True:
            key = get_key().lower()
            if key == 'w': linear = min(linear + 0.5, 1.0); robot.set_linear(linear)
            elif key == 's': linear = max(linear - 0.5, -1.0); robot.set_linear(linear)
            elif key == 'a': angular = max(angular - 0.25, -1.0); robot.set_angular(angular)
            elif key == 'd': angular = min(angular + 0.25, 1.0); robot.set_angular(angular)
            elif key == ' ': linear = 0.0; robot.set_linear(0.0)
            elif key == 'c': angular = 0.0; robot.set_angular(0.0)
            elif key == 'q': break
            else: continue
            print(f"\rlinear={linear:+.2f} angular={angular:+.2f} enc={robot.get_encoder():6d}  ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()

if __name__ == "__main__":
    main()