#!/usr/bin/env python3
import serial, time, wave, numpy as np, sys, pathlib

class ESP32Recorder:
    """
    Keep one serial connection open and grab audio on-demand.

    Protocol (per capture):
        host ->  '1\\n'                # trigger
        esp  ->  'START\\n'
                 '<byte_len>\\n'
                 <raw PCM bytes>
                 'END\\n'
    """

    # ---------- constants you rarely change ----------
    SAMPLE_RATE   = 16_000         # Hz – must match ESP32
    SAMPLE_WIDTH  = 2              # bytes (16-bit)
    CHANNELS      = 1
    PER_CHUNK_SEC = 2              # ESP32 hard-coded clip length
    TRIGGER_CMD   = b'1\n'
    START_TOKEN   = 'START'
    END_TOKEN     = 'END'

    # ---------- constructor ----------
    def __init__(self, port='/dev/ttyACM0', baud=115200, timeout=1):
        try:
            self.ser = serial.Serial(port, baud, timeout=timeout)
        except serial.SerialException as e:
            sys.exit(f"❌ Could not open {port}: {e}")
        time.sleep(2)                   # allow USB-CDC to settle
        self.ser.reset_input_buffer()
        print(f"🔗 Serial ready on {port}")

    # ---------- public API ----------
    def record_chunk(self, duration_s, wav_out=None):
        """
        Collect *duration_s* seconds of audio (multiple 2-s snippets).
        Returns np.int16 array.  If *wav_out* given, also saves to that path.
        """
        target_samples = int(duration_s * self.SAMPLE_RATE)
        chunks, have = [], 0

        while have < target_samples:
            self._trigger_once()
            n_bytes = self._expect_length()
            pcm     = self._read_exact(n_bytes)
            self._eat_end_token()

            pcm_np = np.frombuffer(pcm, dtype=np.int16)
            chunks.append(pcm_np)
            have += pcm_np.size
            print(f"…captured {have/self.SAMPLE_RATE:.2f}/{duration_s:.2f}s")

        audio = np.concatenate(chunks)[:target_samples]  # trim excess

        if wav_out:
            self._save_wav(audio, wav_out)

        return audio

    def close(self):
        self.ser.close()
        print("🔌 Serial closed")

    # ---------- internal helpers ----------
    def _trigger_once(self):
        self.ser.write(self.TRIGGER_CMD)
        line = self._read_line("START")
        if line != self.START_TOKEN:
            raise RuntimeError(f"Expected START, got {line!r}")

    def _expect_length(self):
        line = self._read_line("length")
        if not line.isdigit():
            raise RuntimeError(f"Non-numeric length line: {line!r}")
        return int(line)

    def _eat_end_token(self):
        line = self._read_line("END")
        if line != self.END_TOKEN:
            print(f"[WARN] Expected END, got {line!r} (continuing)")

    def _read_line(self, label):
        while True:
            raw = self.ser.readline()
            if raw:
                try:
                    return raw.decode().strip()
                except UnicodeDecodeError:
                    print(f"[WARN] undecodable {label}: {raw!r}")

    def _read_exact(self, n_bytes):
        buf = bytearray()
        while len(buf) < n_bytes:
            chunk = self.ser.read(n_bytes - len(buf))
            if chunk:
                buf.extend(chunk)
        return bytes(buf)

    def _save_wav(self, pcm_np, path):
        path = pathlib.Path(path)
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.SAMPLE_WIDTH)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(pcm_np.tobytes())
        print(f"💾 Saved {path}")

# ---------------- usage example ----------------
if __name__ == '__main__':
    rec = ESP32Recorder('/dev/ttyACM0', 115200)
    try:
        audio = rec.record_chunk(6, wav_out='test_6s.wav')   # 6-second grab
        print("NumPy audio shape:", audio.shape)
    finally:
        rec.close()
