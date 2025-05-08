#!/usr/bin/env python3
import serial
import wave
import time
import sys

# ——— CONFIG ———
SERIAL_PORT  = '/dev/ttyACM0'  # adjust if needed
BAUD_RATE    = 921600
OUTPUT_WAV   = 'recording.wav'
SAMPLE_RATE  = 16000           # must match your ESP32
SAMPLE_WIDTH = 2               # bytes per sample (16-bit)
CHANNELS     = 1

def debug_dump(buf, label):
    print(f"[DEBUG] {label}: {repr(buf)}")

def main():
    # 1) Open serial port
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"❌ Could not open {SERIAL_PORT}: {e}")
        sys.exit(1)

    time.sleep(2)
    ser.reset_input_buffer()

    # 2) Trigger record
    print("▶ Sending record trigger…")
    ser.write(b'1\n')

    # 3) Wait for START
    while True:
        raw = ser.readline()
        if not raw:
            print("[DEBUG] waiting for START...")
            continue
        debug_dump(raw, "line raw")
        line = raw.decode('utf-8', errors='replace').strip()
        print("→ Parsed:", repr(line))
        if line == 'START':
            print("✅ Got START")
            break

    # 4) Now keep reading until you get a numeric length line
    total_bytes = None
    while total_bytes is None:
        raw_len = ser.readline()
        if not raw_len:
            print("[DEBUG] waiting for length line...")
            continue
        debug_dump(raw_len, "length raw")
        length_line = raw_len.decode('utf-8', errors='replace').strip()
        print("→ Parsed length candidate:", repr(length_line))
        if length_line.isdigit():
            total_bytes = int(length_line)
            print(f"ℹ Will read {total_bytes} bytes of PCM")
        else:
            print("…not a length, ignoring.")

    # 5) Read exactly total_bytes of PCM
    audio = bytearray()
    while len(audio) < total_bytes:
        chunk = ser.read(min(32000, total_bytes - len(audio)))
        if not chunk:
            print(f"[DEBUG] read() timeout at {len(audio)}/{total_bytes}")
            break
        audio.extend(chunk)
        print(f"[DEBUG] received {len(audio)}/{total_bytes}")

    # 6) (Optional) grab the END marker
    raw_end = ser.readline()
    debug_dump(raw_end, "end marker raw")
    end_marker = raw_end.decode('utf-8', errors='replace').strip()
    print("→ End marker parsed:", repr(end_marker))

    ser.close()

    # 7) Write out WAV
    print(f"💾 Writing '{OUTPUT_WAV}'…")
    with wave.open(OUTPUT_WAV, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio)
    print("🏁 Done.")

if __name__ == '__main__':
    main()
