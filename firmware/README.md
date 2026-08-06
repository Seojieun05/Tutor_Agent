# XIAO ESP32S3 Sense — the tutor's eyes

The board is **camera-only**. It connects to `ws://<server>:8765/camera`, says
hello, and answers each `capture_request` with one JPEG. Microphone and speaker
stay on the laptop (the browser page at `http://localhost:8765/`), so the board
never has to run VAD, STT or audio playback — and the laptop is where the sound
comes out anyway.

```text
XIAO ──ws /camera──►  server.py  ◄──ws /browser── laptop (mic + speaker)
        JPEG on request              hint request / hint audio
```

The server pairs them: when the session asking for a hint has no camera of its
own, it borrows a connected one ([tutor/server/camera.py](../tutor/server/camera.py)).

## 1. Assemble

1. Press the **Sense expansion board** onto the XIAO's B2B connector until it
   clicks. To remove it later, slide it off parallel to the board — never pry.
2. Camera ribbon: lift the **black latch** on the FPC connector, insert the
   ribbon with the **contacts facing down**, push it in evenly, then close the
   latch. A half-seated ribbon is the usual cause of `Camera probe failed
   0x105`.
3. Attach the **U.FL antenna** — it ships loose, and Wi-Fi is unreliable without it.

## 2. Arduino IDE setup

| Setting | Value |
|---|---|
| Boards Manager URL | `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json` |
| Board package | **esp32** by Espressif (2.0.14+ or 3.x) |
| Tools > Board | **XIAO_ESP32S3** |
| Tools > PSRAM | **OPI PSRAM** ← not the default, and UXGA needs it |
| Library | **WebSockets** by Markus Sattler (Links2004) |

> Use Links2004's `WebSockets`, not gilmaimon's `ArduinoWebsockets`: the latter
> copies the whole payload into a `std::string` and ignores short TCP writes,
> which truncates or crashes at ~100 KB image sizes. Links2004 sends buffers
> ≥1400 bytes straight from your pointer and loops over partial writes.

## 3. Configure and flash

Edit the settings block at the top of
[`tutor_xiao_camera.ino`](tutor_xiao_camera/tutor_xiao_camera.ino):

```c
#define WIFI_SSID     "your-2.4GHz-ssid"   // 2.4 GHz only — the ESP32-S3 has no 5 GHz radio
#define WIFI_PASSWORD "your-password"
#define SERVER_HOST   "192.168.0.10"       // the machine running server.py
#define SERVER_PORT   8765
```

Flash, then open Serial Monitor at 115200. A healthy boot looks like:

```text
[cam] 준비 완료 (PID 0x26, PSRAM 있음)
[wifi] 연결됨 192.168.0.42
[ws] 연결됨 192.168.0.10:8765/camera
[ws] 서버가 카메라로 등록했습니다
```

and the server logs `camera connected: xiao-1`.

## 4. Reaching the server (server on the laptop)

The board opens a TCP connection to the laptop, so all three pieces sit on one
Wi-Fi network:

```text
        same 2.4 GHz Wi-Fi
XIAO ───────────────────────► laptop:8765 ── server.py
                                   └── browser at http://localhost:8765/
```

1. **Get the address and the settings block** — this prints your laptop's LAN
   IP already filled into the sketch defines:

   ```bash
   python -m tutor.scripts.live_demo
   ```

   The server prints it too, on startup: `camera device (XIAO): ws://<ip>:8765/camera`.

2. **Allow the port through the laptop firewall.** This is the usual reason a
   correctly-flashed board never connects — Windows blocks inbound by default.
   In an **administrator** PowerShell, once:

   ```powershell
   New-NetFirewallRule -DisplayName "Tutor 8765" -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow
   ```

3. **Check reachability** from a phone or another laptop on the same Wi-Fi
   before blaming the board: `nc -vz <laptop-ip> 8765` (or open
   `http://<laptop-ip>:8765/` in a browser — the tutor page should load).

Notes that cost the most time when missed:

- The XIAO and the laptop must be on **2.4 GHz**. A 5 GHz-only SSID is
  invisible to this chip, and many phone hotspots default to 5 GHz.
- The laptop's IP changes when it rejoins a network. If the board stops
  connecting, re-run the preflight and re-flash with the new `SERVER_HOST`.
- **Guest / school Wi-Fi with client isolation** blocks device-to-device
  traffic entirely. A phone hotspot (2.4 GHz) is the reliable fallback.
- If the server runs on a remote host instead, the board cannot use your SSH
  tunnel — a tunnel only exists on the laptop. You would have to open the port
  on that host's firewall.

## 5. Test without hardware

The simulator speaks the identical protocol, so the pairing can be checked (and
the server debugged) before anything is flashed:

```bash
.venv/bin/python -m simulator.camera_device --server ws://localhost:8765 --images simulator/assets/lin_001_wrong_sign.jpg
```

Then talk to the tutor in the browser page: the hint request will be served by
this fake camera exactly as it would by the board.

## 6. "카메라에 다시 보여 줄래요?" and nothing else

That one sentence has two completely different causes, and from the outside they
are identical: either **no frame arrived** (so the VLM was never called), or a
frame arrived and **the read was rejected** as too uncertain. Do not guess —
measure, with the server stopped:

```bash
.venv/bin/python -m tutor.scripts.camera_check
```

It serves only `/camera`, waits for the board, asks for one photo, and prints the
byte count, the transfer time, the dimensions, and where it saved the JPEG — then
sends that exact photo to the VLM and prints what came back. **Open the saved
file.** Half the answer is usually visible in it: the page out of frame, out of
focus, upside down, or too dark.

Reading the result:

| what you see | what it means |
|---|---|
| board never connects | `SERVER_HOST` is not the laptop's LAN IP, or the board is on 5 GHz |
| no frame within the timeout | the transfer is the problem — see the transfer time in the board's serial log |
| frame arrives, `confidence` low | the photo is genuinely hard to read: framing, focus, glare, light |
| frame arrives, JSON looks right | vision is fine; the fault is downstream, so read the server log |

If the board's serial log prints a transfer time above 5 s, the default capture
timeout used to be the whole bug. It is now 15 s and tunable:

```bash
CAPTURE_TIMEOUT_S=25 SAVE_CAPTURES_DIR=data/captures .venv/bin/python server.py
```

`SAVE_CAPTURES_DIR` keeps every frame the server receives during a real lesson,
which is the fastest way to see what the tutor was looking at when it complained.
If the transfer is simply too slow, drop `CAPTURE_SIZE` to `FRAMESIZE_SXGA` in the
sketch — it roughly halves the bytes and stays readable.

## Notes on the image

- **UXGA (1600×1200), quality 10.** Handwriting needs the resolution; SVGA is
  usually too coarse to read reliably. Expect roughly 40–120 KB per frame —
  the serial log prints the real byte count, use that to judge your Wi-Fi.
- **Auto-exposure needs frames, not time.** From cold at UXGA the sensor takes
  seconds to settle, and a `delay()` does not help. The sketch cycles one
  throwaway frame every 3 s so an on-demand capture is exposed correctly, and
  discards two queued frames right before each capture so the photo shows what
  the student is looking at *now*.
- **OV2640 or OV3660.** Newer Sense units ship the OV3660; the sketch detects it
  and applies the vertical flip and saturation fixups it needs.

## Later

Mic and button on the board are deliberately out of scope: endpointing and mp3
playback are far easier on the laptop, and the browser client already does both.
If the XIAO ever needs to be the whole device, it would stream PCM to
`/browser`-style server-side VAD rather than run Silero itself.
