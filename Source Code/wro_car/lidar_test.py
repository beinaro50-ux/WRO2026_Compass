#!/usr/bin/env python3
import sys
import time
from rplidar import RPLidar, RPLidarException

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/rplidar'
BAUD = int(sys.argv[2]) if len(sys.argv) > 2 else 256000

print(f'Opening {PORT} @ {BAUD}')
lidar = RPLidar(PORT, baudrate=BAUD, timeout=3)

try:
    # Clear any state left by a previous crashed run
    lidar.stop()
    lidar.stop_motor()
    time.sleep(0.5)
    lidar.clean_input()

    print('Info:  ', lidar.get_info())
    print('Health:', lidar.get_health())

    lidar.start_motor()
    time.sleep(2)          # let the rotor reach speed before scanning

    for i, scan in enumerate(lidar.iter_scans(max_buf_meas=3000)):
        print(f'{i}: {len(scan)} measurements, '
              f'first={scan[0][1]:.1f}deg {scan[0][2]:.0f}mm')
        if i >= 10:
            break

except RPLidarException as e:
    print('LIDAR error:', e)
except KeyboardInterrupt:
    print('\ninterrupted')
finally:
    try:
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
    except Exception:
        pass
    print('closed cleanly')
