/* 
 * WAV Recorder for Seeed XIAO ESP32S3 Sense 
*/

#include <I2S.h>
#include "FS.h"
#include "SPI.h"
#include <HardwareSerial.h>

// Debug UART settings
#define DEBUG_RX 16
#define DEBUG_TX 17
HardwareSerial DebugSerial(1);

// Recording settings
#define RECORD_TIME   20  // seconds, The maximum value is 240
#define SAMPLE_RATE 16000U
#define SAMPLE_BITS 16
#define WAV_HEADER_SIZE 44
#define VOLUME_GAIN 2

void setup() {
    // USB-C CDC for Pi communication
    Serial.begin(115200);
    
    // Debug UART
    DebugSerial.begin(115200, SERIAL_8N1, DEBUG_RX, DEBUG_TX);
    DebugSerial.println("ESP32-S3 booting...");
    
    // Wait for USB-CDC connection
    while (!Serial) {
        delay(10);
    }
    
    DebugSerial.println("USB-CDC connection established");
    Serial.println("ESP32-S3 USB-C CDC ready");
    
    // Initialize I2S for audio recording
    I2S.setAllPins(-1, 42, 41, -1, -1);
    if (!I2S.begin(PDM_MONO_MODE, SAMPLE_RATE, SAMPLE_BITS)) {
        DebugSerial.println("Failed to initialize I2S!");
        Serial.println("Failed to initialize I2S!");
        while (1) ;
    }
}

void loop() {
    // Check for commands from Pi
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        DebugSerial.print("Received from Pi: ");
        DebugSerial.println(cmd);

        if (cmd == "1") {
            Serial.println("ACK: start recording");
            DebugSerial.println("→ Starting audio recording");
            record_wav();
        }
    }

    // Handle debug UART input
    if (DebugSerial.available()) {
        String dbg = DebugSerial.readStringUntil('\n');
        dbg.trim();
        Serial.println("Debug says: " + dbg);
    }

    delay(100);
}

void record_wav() {
    uint32_t sample_size = 0;
    uint32_t record_size = (SAMPLE_RATE * SAMPLE_BITS / 8) * RECORD_TIME;
    uint8_t *rec_buffer = NULL;
    
    DebugSerial.printf("Ready to start recording...\n");

    // PSRAM malloc for recording
    rec_buffer = (uint8_t *)ps_malloc(record_size);
    if (rec_buffer == NULL) {
        DebugSerial.printf("malloc failed!\n");
        Serial.println("ERROR: Memory allocation failed");
        return;
    }
    DebugSerial.printf("Buffer: %d bytes\n", ESP.getPsramSize() - ESP.getFreePsram());

    // Start recording
    esp_i2s::i2s_read(esp_i2s::I2S_NUM_0, rec_buffer, record_size, &sample_size, portMAX_DELAY);
    if (sample_size == 0) {
        DebugSerial.printf("Record Failed!\n");
        Serial.println("ERROR: Recording failed");
    } else {
        DebugSerial.printf("Record %d bytes\n", sample_size);
        
        // Increase volume
        for (uint32_t i = 0; i < sample_size; i += SAMPLE_BITS/8) {
            (*(uint16_t *)(rec_buffer+i)) <<= VOLUME_GAIN;
        }

        // Send buffer to Pi
        Serial.println("DATA_START:" + String(sample_size));
        DebugSerial.println("Sending audio data to Pi...");
        
        // Send the actual buffer data
        Serial.write(rec_buffer, sample_size);
        
        // Send end marker
        Serial.println("DATA_END");
        DebugSerial.println("Data transfer complete");
    }
    
    // Free buffer after sending
    free(rec_buffer);
}