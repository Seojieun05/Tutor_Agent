// Visual Socratic Tutor — XIAO ESP32S3 Sense as the tutor's eyes.
//
// The board is camera-only and passive: it connects to ws://<server>/camera,
// says hello, and answers each capture_request with ONE high-quality JPEG.
// The microphone and the speaker stay on the laptop (browser page), so this
// sketch never has to do VAD, STT or audio playback — see firmware/README.md.
//
// Wire format (identical to tutor/protocol/frames.py):
//   text frame   : {"type":"EVENT","event":"...","data":{...},"ts":0}
//   binary frame : [0x01][uint32 BE header length][JSON header][JPEG bytes]
//
// Board setup that is easy to get wrong:
//   Tools > Board  : XIAO_ESP32S3
//   Tools > PSRAM  : "OPI PSRAM"   <-- NOT the default; UXGA needs it
// Library: "WebSockets" by Markus Sattler (Links2004), Library Manager.

#include <WiFi.h>
#include <WebSocketsClient.h>
#include "esp_camera.h"
#include "camera_pins.h"

// ---------------------------------------------------------------- settings --
#define WIFI_SSID     "your-2.4GHz-ssid"   // ESP32-S3 is 2.4 GHz only, no 5 GHz
#define WIFI_PASSWORD "your-password"
#define SERVER_HOST   "192.168.0.10"       // the machine running server.py
#define SERVER_PORT   8765
#define SERVER_PATH   "/camera"            // the camera endpoint, not "/"
#define DEVICE_ID     "xiao-1"

// UXGA (1600x1200) reads handwriting reliably; SXGA if Wi-Fi is weak.
static const framesize_t CAPTURE_SIZE = FRAMESIZE_UXGA;
static const int JPEG_QUALITY = 10;  // 0-63, LOWER is better

// The sensor's auto-exposure only converges while frames are actually being
// taken (~6 s from cold at UXGA), and a plain delay() does not help. One
// throwaway frame every few seconds keeps it converged, so an on-demand
// capture is correctly exposed instead of green or black.
static const uint32_t KEEP_AWAKE_MS = 3000;
static const int DISCARD_BEFORE_CAPTURE = 2;

// A white page on a white desk is the worst case for auto-exposure: it meters
// the whole bright scene, opens up, and the page saturates — black ink on white
// paper arrives with almost no contrast left, which is what a real capture from
// this rig looked like. Bias the exposure down and push contrast back up; the
// pixels are the same, the ink is legible.
// Raise AE_LEVEL toward 0 if photos come out too dark for the room.
static const int AE_LEVEL   = -2;  // -2..2, exposure bias
static const int BRIGHTNESS = -1;  // -2..2
static const int CONTRAST   =  1;  // -2..2
static const int SATURATION = -2;  // ink is not a colour; -2 is effectively grey

// ------------------------------------------------------------------ globals --
WebSocketsClient ws;
static bool cameraReady = false;
static uint32_t lastKeepAwake = 0;

// Reserved by the WebSocket library so it can write its own header in front of
// our payload without a second copy (WEBSOCKETS_MAX_HEADER_SIZE).
static const size_t WS_HEADER_RESERVE = 14;

static void led(bool on) { digitalWrite(STATUS_LED_GPIO, on ? LOW : HIGH); }

// ------------------------------------------------------------------- camera --
static bool startCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode    = CAMERA_GRAB_LATEST;  // WHEN_EMPTY returns stale frames

  if (psramFound()) {
    config.frame_size   = CAPTURE_SIZE;
    config.jpeg_quality = JPEG_QUALITY;
    config.fb_count     = 2;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
  } else {
    // Almost certainly "Tools > PSRAM" was left Disabled.
    Serial.println("[cam] PSRAM 없음 → SVGA로 축소 (Tools > PSRAM: OPI PSRAM 확인)");
    config.frame_size   = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count     = 1;
    config.fb_location  = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[cam] init 실패 0x%x — 리본 케이블 래치를 확인하세요\n", err);
    return false;
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s->id.PID == OV3660_PID) {
    // Newer Sense units ship an OV3660, whose image arrives vertically flipped.
    s->set_vflip(s, 1);
  }
  s->set_framesize(s, config.frame_size);  // the streaming examples drop to QVGA here

  // Exposure for paper, not for the room. Applied to both sensors: the page is
  // the subject on every unit, and the OV3660's own tuning was the same idea
  // pointed the wrong way (it used to raise brightness, blowing the page out).
  s->set_ae_level(s, AE_LEVEL);
  s->set_brightness(s, BRIGHTNESS);
  s->set_contrast(s, CONTRAST);
  s->set_saturation(s, SATURATION);

  // Cycle frames so auto-exposure converges before the first real request.
  for (int i = 0; i < 8; i++) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
    delay(120);
  }
  Serial.printf("[cam] 준비 완료 (PID 0x%x, PSRAM %s)\n",
                s->id.PID, psramFound() ? "있음" : "없음");
  return true;
}

// ------------------------------------------------------------ wire protocol --
static void sendEvent(const char *event, const String &dataJson) {
  String msg = String("{\"type\":\"EVENT\",\"event\":\"") + event +
               "\",\"data\":" + dataJson + ",\"ts\":0}";
  ws.sendTXT(msg);
}

// Minimal extractor for our own server's JSON — no parser dependency.
static String jsonString(const char *json, const char *key) {
  String needle = String("\"") + key + "\":\"";
  const char *at = strstr(json, needle.c_str());
  if (!at) return String();
  at += needle.length();
  const char *end = strchr(at, '"');
  return end ? String(at).substring(0, end - at) : String();
}

// One binary frame: [0x01][uint32 BE header length][JSON header][JPEG].
// The JPEG is copied once into PSRAM because our framing needs a prefix in
// front of it and the driver's buffer has no headroom. headerToPayload lets
// the WebSocket library write its own header into WS_HEADER_RESERVE instead
// of allocating a second copy of the whole image.
static bool sendImage(camera_fb_t *fb, const String &captureId) {
  String header = String("{\"capture_id\":\"") + captureId +
                  "\",\"format\":\"jpeg\",\"width\":" + fb->width +
                  ",\"height\":" + fb->height + ",\"seq\":0}";
  const size_t headerLen = header.length();
  const size_t frameLen = 1 + 4 + headerLen + fb->len;

  uint8_t *buf = (uint8_t *) ps_malloc(WS_HEADER_RESERVE + frameLen);
  if (!buf) {
    Serial.println("[ws] PSRAM 할당 실패");
    return false;
  }
  uint8_t *p = buf + WS_HEADER_RESERVE;
  p[0] = 0x01;  // IMAGE
  p[1] = (headerLen >> 24) & 0xFF;   // uint32 big-endian
  p[2] = (headerLen >> 16) & 0xFF;
  p[3] = (headerLen >> 8) & 0xFF;
  p[4] = headerLen & 0xFF;
  memcpy(p + 5, header.c_str(), headerLen);
  memcpy(p + 5 + headerLen, fb->buf, fb->len);

  bool ok = ws.sendBIN(buf, frameLen, true);
  free(buf);
  return ok;
}

static void handleCaptureRequest(const String &captureId) {
  if (!cameraReady) {
    sendEvent("capture_failed", String("{\"capture_id\":\"") + captureId + "\"}");
    return;
  }
  led(true);
  // The oldest queued frame predates the request; drop a couple so the photo
  // shows what the student is looking at now.
  for (int i = 0; i < DISCARD_BEFORE_CAPTURE; i++) {
    camera_fb_t *stale = esp_camera_fb_get();
    if (stale) esp_camera_fb_return(stale);
  }
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[cam] 촬영 실패");
    sendEvent("capture_failed", String("{\"capture_id\":\"") + captureId + "\"}");
    led(false);
    return;
  }
  uint32_t started = millis();
  bool ok = sendImage(fb, captureId);
  Serial.printf("[cam] %s: %u bytes %ux%u → %s (%lu ms)\n",
                captureId.c_str(), (unsigned) fb->len, fb->width, fb->height,
                ok ? "전송" : "전송 실패", millis() - started);
  esp_camera_fb_return(fb);  // must happen even on failure, or the next grab starves
  if (!ok) sendEvent("capture_failed", String("{\"capture_id\":\"") + captureId + "\"}");
  led(false);
}

static void onWsEvent(WStype_t type, uint8_t *payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED:
      Serial.printf("[ws] 연결됨 %s:%d%s\n", SERVER_HOST, SERVER_PORT, SERVER_PATH);
      sendEvent("hello", String("{\"device_id\":\"") + DEVICE_ID +
                             "\",\"fw_version\":\"1.0\",\"proto\":1,\"caps\":[\"camera\"]}");
      break;
    case WStype_DISCONNECTED:
      Serial.println("[ws] 연결 끊김 — 재연결 대기");
      led(false);
      break;
    case WStype_TEXT: {
      const char *json = (const char *) payload;  // library NUL-terminates it
      if (strstr(json, "\"capture_request\"")) {
        handleCaptureRequest(jsonString(json, "capture_id"));
      } else if (strstr(json, "\"hello_ack\"")) {
        Serial.println("[ws] 서버가 카메라로 등록했습니다");
      }
      break;
    }
    default:
      break;
  }
}

// -------------------------------------------------------------------- setup --
void setup() {
  Serial.begin(115200);
  delay(300);
  pinMode(STATUS_LED_GPIO, OUTPUT);
  led(false);

  cameraReady = startCamera();

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // modem sleep stalls large uploads mid-transfer
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("[wifi] %s 접속 중", WIFI_SSID);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\n[wifi] 연결됨 %s\n", WiFi.localIP().toString().c_str());

  ws.begin(SERVER_HOST, SERVER_PORT, SERVER_PATH);
  ws.onEvent(onWsEvent);
  ws.setReconnectInterval(5000);
  ws.enableHeartbeat(15000, 3000, 2);
}

void loop() {
  ws.loop();

  if (cameraReady && millis() - lastKeepAwake > KEEP_AWAKE_MS) {
    lastKeepAwake = millis();
    camera_fb_t *fb = esp_camera_fb_get();  // keeps auto-exposure converged
    if (fb) esp_camera_fb_return(fb);
  }
}
