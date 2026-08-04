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

## 4. Reaching the server

The board must be able to open a TCP connection to the server, which is the
part that usually bites:

- **Server on the same LAN** — use its LAN IP. Nothing else to do.
- **Server on a remote/cloud host** (this project's usual setup) — the XIAO
  cannot use your SSH tunnel; a tunnel only exists on the laptop. Either open
  the port to the board's network (Azure NSG / security group inbound rule for
  8765), or run the server on the laptop instead, or put a small relay on the
  LAN. Check reachability first:
  `nc -vz <server-ip> 8765` from a laptop on the *same Wi-Fi as the board*.
- The XIAO and the phone/laptop hotspot must be on **2.4 GHz**. A 5 GHz-only
  SSID is invisible to this chip.

## 5. Test without hardware

The simulator speaks the identical protocol, so the pairing can be checked (and
the server debugged) before anything is flashed:

```bash
.venv/bin/python -m simulator.camera_device --server ws://localhost:8765 --images simulator/assets/lin_001_wrong_sign.jpg
```

Then talk to the tutor in the browser page: the hint request will be served by
this fake camera exactly as it would by the board.

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
