// XIAO ESP32S3 Sense — OV2640/OV3660 pin map.
//
// These are the CAMERA_MODEL_XIAO_ESP32S3 values from the Espressif Arduino
// core (libraries/ESP32/examples/Camera/CameraWebServer/camera_pins.h).
// Note the inversion when they are assigned: pin_d7 = Y9 ... pin_d0 = Y2.
// There is no camera flash LED on this board; the user LED is GPIO 21.

#pragma once

#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  10
#define SIOD_GPIO_NUM  40  // SDA
#define SIOC_GPIO_NUM  39  // SCL

#define Y9_GPIO_NUM    48
#define Y8_GPIO_NUM    11
#define Y7_GPIO_NUM    12
#define Y6_GPIO_NUM    14
#define Y5_GPIO_NUM    16
#define Y4_GPIO_NUM    18
#define Y3_GPIO_NUM    17
#define Y2_GPIO_NUM    15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM  47
#define PCLK_GPIO_NUM  13

#define STATUS_LED_GPIO 21  // on-board user LED, active LOW
