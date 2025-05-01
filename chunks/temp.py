#!/usr/bin/env python3
import wave, numpy as np, matplotlib.pyplot as plt, sys

fname = "capture.mp3" if len(sys.argv) < 2 else sys.argv[1]

with wave.open(fname, "rb") as w:
    n_frames  = w.getnframes()
    framerate = w.getframerate()
    data = np.frombuffer(w.readframes(n_frames), dtype=np.int16)

t = np.arange(n_frames) / framerate
plt.figure(figsize=(10, 3))
plt.plot(t, data, linewidth=0.5)
plt.title(f"Waveform of {fname}  ({framerate} Hz, {n_frames/framerate:.2f} s)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
