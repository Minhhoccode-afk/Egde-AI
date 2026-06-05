#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include "WSEN_ITDS.h"
#include <Adafruit_INA219.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <math.h>

// TensorFlow Lite Micro
#include <TensorFlowLite_ESP32.h>
#include "model_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

// --- KHAI BÁO CHÂN PHẦN CỨNG (XIAO ESP32-S3) ---\r
#define ENA 1
#define IN1 2
#define IN2 3
#define LED_PIN 43
#define BUZ_PIN 44

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1

// --- THÔNG TIN MẠNG WI-FI PHÒNG LAB ---
const char* ssid = "happy123_IoT";
const char* password = "0915321223";

// --- KHỞI TẠO ĐỐI TƯỢNG ---
Sensor_ITDS sensor;
Adafruit_INA219 ina219;
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
AsyncWebServer server(80);

// --- BIẾN TOÀN CỤC CHIA SẺ GIỮA 2 CORE ---
SemaphoreHandle_t dataMutex;
volatile int current_state = 0;       // 0: Normal, 1: Warning, 2: Danger
volatile float max_prob = 0.0;        // Xác suất rủi ro lớn nhất (%)
volatile bool motor_running = false;  // Trạng thái động cơ chạy/dừng
volatile bool system_halted = false;  // Khóa an toàn hệ thống (Latching Fault)
int warning_count = 0;                // Bộ lọc Debounce cho trạng thái Warning

// Sensor snapshot for Web/OLED/Serial
volatile float acc_x_mg = 0.0f;
volatile float acc_y_mg = 0.0f;
volatile float acc_z_mg = 0.0f;
volatile float bus_voltage_v = 0.0f;
volatile float current_ma = 0.0f;
volatile float current_baseline_ma = 0.0f;
volatile bool startup_boost_active = false;
volatile uint32_t startup_boost_until_ms = 0;
volatile uint32_t motor_start_ms = 0;
volatile uint32_t led_last_toggle_ms = 0;
volatile bool led_level = false;

// On-device XAI (feature ablation style contribution)
volatile float xai_contrib_x = 0.0f;
volatile float xai_contrib_y = 0.0f;
volatile float xai_contrib_z = 0.0f;
volatile float xai_contrib_current = 0.0f;
volatile float xai_contrib_voltage = 0.0f;
constexpr float XAI_EMA_ALPHA = 0.2f;

// Adaptive thresholding (self-learning during startup grace: mu + k*sigma)
struct OnlineStats {
  uint32_t n = 0;
  float mean = 0.0f;
  float M2 = 0.0f;

  void reset() {
    n = 0;
    mean = 0.0f;
    M2 = 0.0f;
  }

  void push(float x) {
    n++;
    float delta = x - mean;
    mean += delta / (float)n;
    float delta2 = x - mean;
    M2 += delta * delta2;
  }

  float stddev() const {
    if (n < 2) return 0.0f;
    return sqrtf(M2 / (float)(n - 1));
  }

  float thresholdSigma(float k) const {
    return mean + k * stddev();
  }
};

OnlineStats stats_curr;
OnlineStats stats_vib;
OnlineStats stats_ax;
OnlineStats stats_ay;
OnlineStats stats_az;
OnlineStats stats_busv;

volatile bool adaptive_learning = false;
volatile bool adaptive_ready = false;
volatile float adapt_mu_curr = 0.0f;
volatile float adapt_mu_vib = 0.0f;
volatile float adapt_warn_curr = 0.0f;
volatile float adapt_dang_curr = 0.0f;
volatile float adapt_warn_vib = 0.0f;
volatile float adapt_dang_vib = 0.0f;

// Sliding window 50 x 5 for TinyML
constexpr int WINDOW_SIZE = 50;
constexpr int FEATURE_SIZE = 5;
float feature_window[WINDOW_SIZE][FEATURE_SIZE];
int window_index = 0;
bool window_ready = false;

// TensorFlow Lite Micro globals
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter *error_reporter = &micro_error_reporter;
const tflite::Model *model = nullptr;
tflite::AllOpsResolver resolver;
constexpr int kTensorArenaSize = 50 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input_tensor = nullptr;
TfLiteTensor *output_tensor = nullptr;

constexpr int PWM_RUN = 185;
constexpr int PWM_START_BOOST = 255;
constexpr uint32_t START_BOOST_MS = 650;
constexpr uint32_t STARTUP_GRACE_MS = 1500;

// --- GIAO DIỆN WEB UI CAO CẤP CHUẨN CÔNG NGHIỆP (OFFLINE HOÀN TOÀN) ---
const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Edge AI Industrial Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #0f0f12; color: #d1d1d6; margin: 0; padding: 20px; }
        .wrapper { max-width: 1100px; margin: 0 auto; }
        header { border-bottom: 2px solid #2c2c35; padding-bottom: 15px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center; }
        h1 { margin: 0; font-size: 24px; color: #ffffff; letter-spacing: 0.5px; text-transform: uppercase; }
        .system-tag { background: #2c2c35; padding: 6px 12px; border-radius: 4px; font-size: 12px; font-weight: bold; color: #00e676; border: 1px solid #3a3a4c; }
        .grid-layout { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; margin-bottom: 25px; }
        .card { background: #17171e; border-radius: 8px; padding: 20px; border: 1px solid #22222b; box-shadow: 0 10px 20px rgba(0,0,0,0.3); position: relative; }
        .card-header { font-size: 13px; text-transform: uppercase; color: #8e8e93; font-weight: 600; letter-spacing: 1px; margin-bottom: 15px; }
        .control-panel { display: flex; flex-direction: column; justify-content: center; gap: 12px; }
        .btn { border: none; padding: 14px; font-size: 15px; font-weight: 700; border-radius: 6px; cursor: pointer; text-transform: uppercase; transition: all 0.2s ease; }
        .btn-start { background: #00c853; color: #fff; }
        .btn-start:hover { background: #00e676; box-shadow: 0 0 15px rgba(0,230,118,0.3); }
        .btn-stop { background: #ff3b30; color: #fff; }
        .btn-stop:hover { background: #ff453a; box-shadow: 0 0 15px rgba(255,69,58,0.3); }
        .btn-reset { background: #ff9f0a; color: #000; }
        .btn-reset:hover { background: #ffb340; box-shadow: 0 0 15px rgba(255,159,10,0.3); }
        
        /* State Status Panel Dynamic Borders */
        .border-normal { border-left: 6px solid #00c853; }
        .border-warning { border-left: 6px solid #ffd60a; }
        .border-danger { border-left: 6px solid #ff3b30; }
        
        .status-text { font-size: 36px; font-weight: 800; margin: 10px 0; letter-spacing: -0.5px; }
        .text-normal { color: #00c853; }
        .text-warning { color: #ffd60a; animation: pulse 1.5s infinite; }
        .text-danger { color: #ff3b30; animation: blink 1s infinite; }
        
        @keyframes pulse { 0% { opacity: 0.6; } 50% { opacity: 1; } 100% { opacity: 0.6; } }
        @keyframes blink { 0% { opacity: 0.2; } 50% { opacity: 1; } 100% { opacity: 0.2; } }
        
        .canvas-container { display: flex; justify-content: center; align-items: center; height: 160px; }
        canvas { border-radius: 4px; }
    </style>
</head>
<body>
    <div class="wrapper">
        <header>
            <h1>ESP32 Edge AI Monitoring System</h1>
            <div class="system-tag" id="system_mode">SYSTEM READY</div>
        </header>

        <div class="grid-layout">
            <div class="card">
                <div class="card-header">Interactive Control Panel</div>
                <div class="control-panel">
                    <button class="btn btn-start" onclick="sendCmd('/start')">Start Motor</button>
                    <button class="btn btn-stop" onclick="sendCmd('/stop')">Stop Motor</button>
                    <button class="btn btn-reset" onclick="sendCmd('/reset')">Reset Alarm</button>
                </div>
            </div>

            <div id="status_card" class="card border-normal">
                <div class="card-header">AI Classification Status</div>
                <div id="ai_status" class="status-text text-normal">NORMAL</div>
                <div style="font-size:14px; color:#8e8e93;">
                    Motor Status: <span id="motor_status" style="font-weight:bold; color:#fff;">STOPPED</span>
                </div>
            </div>

            <div class="card">
                <div class="card-header">AI Risk Probability Gauge</div>
                <div class="canvas-container">
                    <canvas id="gaugeCanvas" width="160" height="160"></canvas>
                </div>
            </div>
        </div>

        <div class="grid-layout">
            <div class="card">
                <div class="card-header">Power Sensor (INA219)</div>
                <div style="font-size:16px; line-height:1.8;">
                    Bus Voltage: <span id="bus_v" style="font-weight:bold; color:#fff;">0.00</span> V<br>
                    Current: <span id="cur_ma" style="font-weight:bold; color:#fff;">0.00</span> mA
                </div>
            </div>
            <div class="card">
                <div class="card-header">Vibration Sensor (WSEN-ITDS)</div>
                <div style="font-size:16px; line-height:1.8;">
                    X: <span id="ax" style="font-weight:bold; color:#fff;">0.0</span> mg<br>
                    Y: <span id="ay" style="font-weight:bold; color:#fff;">0.0</span> mg<br>
                    Z: <span id="az" style="font-weight:bold; color:#fff;">0.0</span> mg
                </div>
            </div>
        </div>

        <div class="card" style="margin-bottom:0;">
            <div class="card-header">Real-Time Risk Probability Trend (%)</div>
            <div style="width:100%; height:200px; margin-top:10px;">
                <canvas id="chartCanvas" style="width:100%; height:100%; background:#111116;"></canvas>
            </div>
        </div>

        <div class="card" style="margin-top:20px;">
            <div class="card-header">Adaptive AI Thresholds (Self-Learned)</div>
            <div style="font-size:14px; line-height:1.8; color:#8e8e93;">
                Status: <span id="adapt_status" style="font-weight:bold; color:#fff;">WAITING</span><br>
                Current Warning/Danger: <span id="thr_warn_i" style="color:#ffd60a;">--</span> / <span id="thr_dang_i" style="color:#ff3b30;">--</span> mA<br>
                Vibration Warning/Danger: <span id="thr_warn_v" style="color:#ffd60a;">--</span> / <span id="thr_dang_v" style="color:#ff3b30;">--</span> mg-dev
            </div>
        </div>

        <div class="card" style="margin-top:20px;">
            <div class="card-header">On-Device XAI (Feature Contribution %)</div>
            <div style="display:grid; grid-template-columns: 130px 1fr 55px; gap:8px; align-items:center;">
                <div>X-Axis</div><div style="background:#22222b; height:10px; border-radius:6px;"><div id="bar_x" style="height:10px; width:0%; background:#29b6f6; border-radius:6px;"></div></div><div id="xai_x">0%</div>
                <div>Y-Axis</div><div style="background:#22222b; height:10px; border-radius:6px;"><div id="bar_y" style="height:10px; width:0%; background:#42a5f5; border-radius:6px;"></div></div><div id="xai_y">0%</div>
                <div>Z-Axis</div><div style="background:#22222b; height:10px; border-radius:6px;"><div id="bar_z" style="height:10px; width:0%; background:#7e57c2; border-radius:6px;"></div></div><div id="xai_z">0%</div>
                <div>Current</div><div style="background:#22222b; height:10px; border-radius:6px;"><div id="bar_i" style="height:10px; width:0%; background:#ef5350; border-radius:6px;"></div></div><div id="xai_i">0%</div>
                <div>Voltage</div><div style="background:#22222b; height:10px; border-radius:6px;"><div id="bar_v" style="height:10px; width:0%; background:#ffa726; border-radius:6px;"></div></div><div id="xai_v">0%</div>
            </div>
        </div>
    </div>

    <script>
        let probabilities = Array(50).fill(0);
        
        function sendCmd(endpoint) {
            fetch(endpoint, { method: 'POST' });
        }

        function drawGauge(val, state) {
            const canvas = document.getElementById('gaugeCanvas');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, 160, 160);
            
            let color = "#00c853";
            if(state === 1) color = "#ffd60a";
            if(state === 2) color = "#ff3b30";

            ctx.beginPath();
            ctx.arc(80, 80, 65, 0.75 * Math.PI, 2.25 * Math.PI);
            ctx.strokeStyle = "#22222b";
            ctx.lineWidth = 12;
            ctx.stroke();

            let endAngle = 0.75 * Math.PI + (val / 100) * 1.5 * Math.PI;
            ctx.beginPath();
            ctx.arc(80, 80, 65, 0.75 * Math.PI, endAngle);
            ctx.strokeStyle = color;
            ctx.lineWidth = 12;
            ctx.lineCap = "round";
            ctx.stroke();

            ctx.fillStyle = "#ffffff";
            ctx.font = "bold 26px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText(val.toFixed(1) + "%", 80, 85);
            ctx.fillStyle = "#8e8e93";
            ctx.font = "11px sans-serif";
            ctx.fillText("RISK INDEX", 80, 105);
        }

        function drawChart() {
            const canvas = document.getElementById('chartCanvas');
            const ctx = canvas.getContext('2d');
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight;
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.beginPath();
            ctx.strokeStyle = "#3a3a4c";
            ctx.lineWidth = 1;
            for(let i=1; i<4; i++) {
                let y = (canvas.height / 4) * i;
                ctx.moveTo(0, y); ctx.lineTo(canvas.width, y);
            }
            ctx.stroke();

            ctx.beginPath();
            ctx.lineWidth = 3;
            ctx.strokeStyle = "#00e676";
            
            let step = canvas.width / (probabilities.length - 1);
            for(let i=0; i<probabilities.length; i++) {
                let x = i * step;
                let y = canvas.height - (probabilities[i] / 100) * canvas.height * 0.9 - 10;
                if(i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        setInterval(() => {
            fetch('/data')
                .then(res => res.json())
                .then(data => {
                    const statusCard = document.getElementById('status_card');
                    const aiStatus = document.getElementById('ai_status');
                    const motorStatus = document.getElementById('motor_status');
                    const sysMode = document.getElementById('system_mode');
                    document.getElementById('bus_v').innerText = Number(data.bus_v).toFixed(2);
                    document.getElementById('cur_ma').innerText = Number(data.cur_ma).toFixed(2);
                    document.getElementById('ax').innerText = Number(data.ax).toFixed(1);
                    document.getElementById('ay').innerText = Number(data.ay).toFixed(1);
                    document.getElementById('az').innerText = Number(data.az).toFixed(1);
                    document.getElementById('xai_x').innerText = Number(data.xai_x).toFixed(1) + "%";
                    document.getElementById('xai_y').innerText = Number(data.xai_y).toFixed(1) + "%";
                    document.getElementById('xai_z').innerText = Number(data.xai_z).toFixed(1) + "%";
                    document.getElementById('xai_i').innerText = Number(data.xai_i).toFixed(1) + "%";
                    document.getElementById('xai_v').innerText = Number(data.xai_v).toFixed(1) + "%";
                    if (data.adapt_learning === 1) {
                        document.getElementById('adapt_status').innerText = "LEARNING (Startup Grace)";
                        document.getElementById('adapt_status').style.color = "#ffd60a";
                    } else if (data.adapt_ready === 1) {
                        document.getElementById('adapt_status').innerText = "CALIBRATED";
                        document.getElementById('adapt_status').style.color = "#00c853";
                    } else {
                        document.getElementById('adapt_status').innerText = "WAITING FOR MOTOR START";
                        document.getElementById('adapt_status').style.color = "#fff";
                    }
                    document.getElementById('thr_warn_i').innerText = data.adapt_ready ? Number(data.thr_warn_i).toFixed(1) : "--";
                    document.getElementById('thr_dang_i').innerText = data.adapt_ready ? Number(data.thr_dang_i).toFixed(1) : "--";
                    document.getElementById('thr_warn_v').innerText = data.adapt_ready ? Number(data.thr_warn_v).toFixed(1) : "--";
                    document.getElementById('thr_dang_v').innerText = data.adapt_ready ? Number(data.thr_dang_v).toFixed(1) : "--";
                    document.getElementById('bar_x').style.width = Number(data.xai_x).toFixed(1) + "%";
                    document.getElementById('bar_y').style.width = Number(data.xai_y).toFixed(1) + "%";
                    document.getElementById('bar_z').style.width = Number(data.xai_z).toFixed(1) + "%";
                    document.getElementById('bar_i').style.width = Number(data.xai_i).toFixed(1) + "%";
                    document.getElementById('bar_v').style.width = Number(data.xai_v).toFixed(1) + "%";

                    probabilities.push(data.prob);
                    probabilities.shift();

                    if (data.halted === 1) {
                        aiStatus.innerText = "SYSTEM HALTED";
                        aiStatus.className = "status-text text-danger";
                        statusCard.className = "card border-danger";
                        motorStatus.innerText = "LOCKED (CRITICAL Fault)";
                        motorStatus.style.color = "#ff3b30";
                        sysMode.innerText = "FAULT DETECTED";
                        sysMode.style.color = "#ff3b30";
                    } else {
                        sysMode.innerText = "SYSTEM READY";
                        sysMode.style.color = "#00e676";
                        if (data.state === 0) {
                            aiStatus.innerText = "NORMAL";
                            aiStatus.className = "status-text text-normal";
                            statusCard.className = "card border-normal";
                        } else if (data.state === 1) {
                            aiStatus.innerText = "WARNING";
                            aiStatus.className = "status-text text-warning";
                            statusCard.className = "card border-warning";
                        } else {
                            aiStatus.innerText = "DANGER";
                            aiStatus.className = "status-text text-danger";
                            statusCard.className = "card border-danger";
                        }
                        motorStatus.innerText = data.motor === 1 ? "RUNNING" : "STOPPED";
                        motorStatus.style.color = data.motor === 1 ? "#00c853" : "#fff";
                    }

                    drawGauge(data.prob, data.state);
                    drawChart();
                });
        }, 400);
    </script>
</body>
</html>
)rawliteral";

void applyMotorOutputLocked() {
  if (system_halted || !motor_running) {
    startup_boost_active = false;
    analogWrite(ENA, 0);
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    return;
  }

  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  if (startup_boost_active) {
    if (millis() < startup_boost_until_ms) {
      analogWrite(ENA, PWM_START_BOOST);
    } else {
      startup_boost_active = false;
      analogWrite(ENA, PWM_RUN);
    }
  } else {
    analogWrite(ENA, PWM_RUN);
  }
}

void resetAdaptiveCalibration() {
  stats_curr.reset();
  stats_vib.reset();
  stats_ax.reset();
  stats_ay.reset();
  stats_az.reset();
  stats_busv.reset();
  adaptive_learning = true;
  adaptive_ready = false;
  adapt_mu_curr = 0.0f;
  adapt_mu_vib = 0.0f;
  adapt_warn_curr = 0.0f;
  adapt_dang_curr = 0.0f;
  adapt_warn_vib = 0.0f;
  adapt_dang_vib = 0.0f;
}

void finalizeAdaptiveCalibration() {
  if (stats_curr.n < 10 || stats_vib.n < 10) {
    return;
  }

  adapt_mu_curr = stats_curr.mean;
  adapt_mu_vib = stats_vib.mean;
  // Z-score: Warning ~ 95th percentile (mu + 2sigma), Danger ~ 99.7% (mu + 3sigma)
  adapt_warn_curr = stats_curr.thresholdSigma(2.0f);
  adapt_dang_curr = stats_curr.thresholdSigma(3.0f);
  adapt_warn_vib = stats_vib.thresholdSigma(2.0f);
  adapt_dang_vib = stats_vib.thresholdSigma(3.0f);

  adaptive_learning = false;
  adaptive_ready = true;

  Serial.println("[ADAPTIVE] Calibration complete (mu + k*sigma):");
  Serial.printf("  Current  mu=%.1f warn=%.1f danger=%.1f mA\n",
                adapt_mu_curr, adapt_warn_curr, adapt_dang_curr);
  Serial.printf("  Vibration mu=%.1f warn=%.1f danger=%.1f mg-dev\n",
                adapt_mu_vib, adapt_warn_vib, adapt_dang_vib);
}

float computeVibrationDeviation(float ax, float ay, float az) {
  float refX = stats_ax.n > 0 ? stats_ax.mean : ax;
  float refY = stats_ay.n > 0 ? stats_ay.mean : ay;
  float refZ = stats_az.n > 0 ? stats_az.mean : az;
  return fabsf(ax - refX) + fabsf(ay - refY) + fabsf(az - refZ);
}

void updateAdaptiveCalibration(float ax, float ay, float az, float curr, float busv) {
  stats_ax.push(ax);
  stats_ay.push(ay);
  stats_az.push(az);
  stats_curr.push(curr);
  stats_busv.push(busv);
  stats_vib.push(computeVibrationDeviation(ax, ay, az));
}

int evaluateAdaptiveAnomaly(float ax, float ay, float az, float curr) {
  if (!adaptive_ready) {
    return 0;
  }

  float vib_dev = computeVibrationDeviation(ax, ay, az);
  bool danger = (curr >= adapt_dang_curr) || (vib_dev >= adapt_dang_vib);
  bool warning = (curr >= adapt_warn_curr) || (vib_dev >= adapt_warn_vib);

  if (danger) return 2;
  if (warning) return 1;
  return 0;
}

void startMotorLocked() {
  motor_running = true;
  current_state = 0;
  warning_count = 0;
  max_prob = 0.0f;
  current_baseline_ma = current_ma;
  resetAdaptiveCalibration();
  startup_boost_active = true;
  motor_start_ms = millis();
  startup_boost_until_ms = millis() + START_BOOST_MS;
  applyMotorOutputLocked();
}

void stopMotorLocked() {
  motor_running = false;
  warning_count = 0;
  current_state = 0;
  max_prob = 0.0f;
  digitalWrite(BUZ_PIN, LOW);
  applyMotorOutputLocked();
}

void computeOnDeviceXAI(float ax, float ay, float az, float curr, float busv) {
  // Use self-learned references when calibrated; fallback to live baseline otherwise.
  float refX = (stats_ax.n > 0) ? stats_ax.mean : ax;
  float refY = (stats_ay.n > 0) ? stats_ay.mean : ay;
  float refZ = (stats_az.n > 0) ? stats_az.mean : az;
  float refI = adaptive_ready ? adapt_mu_curr : (motor_running ? current_baseline_ma : curr);
  float refV = (stats_busv.n > 0) ? stats_busv.mean : busv;

  float sx = fabsf(ax - refX);
  float sy = fabsf(ay - refY);
  float sz = fabsf(az - refZ);
  float si = fabsf(curr - refI);
  float sv = fabsf(busv - refV);

  float total = sx + sy + sz + si + sv + 1e-6f;
  float raw_x = (sx / total) * 100.0f;
  float raw_y = (sy / total) * 100.0f;
  float raw_z = (sz / total) * 100.0f;
  float raw_i = (si / total) * 100.0f;
  float raw_v = (sv / total) * 100.0f;

  // EMA smoothing for stable XAI bars on Web dashboard.
  xai_contrib_x = (1.0f - XAI_EMA_ALPHA) * xai_contrib_x + XAI_EMA_ALPHA * raw_x;
  xai_contrib_y = (1.0f - XAI_EMA_ALPHA) * xai_contrib_y + XAI_EMA_ALPHA * raw_y;
  xai_contrib_z = (1.0f - XAI_EMA_ALPHA) * xai_contrib_z + XAI_EMA_ALPHA * raw_z;
  xai_contrib_current = (1.0f - XAI_EMA_ALPHA) * xai_contrib_current + XAI_EMA_ALPHA * raw_i;
  xai_contrib_voltage = (1.0f - XAI_EMA_ALPHA) * xai_contrib_voltage + XAI_EMA_ALPHA * raw_v;
}

void updateStatusLedLocked() {
  uint32_t now = millis();
  uint32_t interval = 1000;
  bool forcedLevel = false;
  bool forceMode = false;

  if (WiFi.status() != WL_CONNECTED) {
    interval = 220;  // waiting Wi-Fi
  } else if (system_halted || current_state == 2) {
    interval = 120;  // danger/halted blink fast
  } else if (current_state == 1) {
    interval = 380;  // warning blink medium
  } else if (!motor_running) {
    forceMode = true;
    forcedLevel = true;  // ready steady ON
  } else {
    interval = 900; // running normal slow heartbeat
  }

  if (forceMode) {
    led_level = forcedLevel;
    digitalWrite(LED_PIN, led_level ? HIGH : LOW);
    return;
  }

  if (now - led_last_toggle_ms >= interval) {
    led_last_toggle_ms = now;
    led_level = !led_level;
    digitalWrite(LED_PIN, led_level ? HIGH : LOW);
  }
}

void pushFeatureWindow(float ax, float ay, float az, float curr, float busv) {
  if (window_index < WINDOW_SIZE) {
    feature_window[window_index][0] = ax;
    feature_window[window_index][1] = ay;
    feature_window[window_index][2] = az;
    feature_window[window_index][3] = curr;
    feature_window[window_index][4] = busv;
    window_index++;
    if (window_index >= WINDOW_SIZE) {
      window_ready = true;
    }
  } else {
    for (int i = 1; i < WINDOW_SIZE; i++) {
      for (int j = 0; j < FEATURE_SIZE; j++) {
        feature_window[i - 1][j] = feature_window[i][j];
      }
    }
    feature_window[WINDOW_SIZE - 1][0] = ax;
    feature_window[WINDOW_SIZE - 1][1] = ay;
    feature_window[WINDOW_SIZE - 1][2] = az;
    feature_window[WINDOW_SIZE - 1][3] = curr;
    feature_window[WINDOW_SIZE - 1][4] = busv;
  }
}

void updateAIState(int predicted_state, float predicted_prob) {
  if (system_halted) {
    current_state = 2;
    max_prob = 100.0f;
    digitalWrite(BUZ_PIN, HIGH);
    return;
  }

  if (predicted_state == 2) {
    current_state = 2;
    max_prob = predicted_prob;
    system_halted = true;
    motor_running = false;
    applyMotorOutputLocked();
    digitalWrite(BUZ_PIN, HIGH);
    Serial.println("[SAFETY] DANGER detected -> SYSTEM HALTED (latched). Press RESET.");
    return;
  }

  if (predicted_state == 1) {
    warning_count++;
    if (warning_count >= 4) {
      current_state = 1;
      max_prob = predicted_prob;
      digitalWrite(BUZ_PIN, HIGH);
    } else if (current_state == 0) {
      max_prob = predicted_prob;
    }
  } else {
    warning_count = 0;
    current_state = 0;
    max_prob = predicted_prob;
    digitalWrite(BUZ_PIN, LOW);
  }
}

float dequantizeTensorValue(const TfLiteTensor *tensor, int idx) {
  if (tensor->type == kTfLiteInt8) {
    const int8_t q = tensor->data.int8[idx];
    return (q - tensor->params.zero_point) * tensor->params.scale;
  }
  if (tensor->type == kTfLiteUInt8) {
    const uint8_t q = tensor->data.uint8[idx];
    return (q - tensor->params.zero_point) * tensor->params.scale;
  }
  if (tensor->type == kTfLiteFloat32) {
    return tensor->data.f[idx];
  }
  return 0.0f;
}

void runInferenceAndUpdateState() {
  bool in_startup_grace = motor_running && (millis() - motor_start_ms < STARTUP_GRACE_MS);

  if (in_startup_grace) {
    updateAdaptiveCalibration(acc_x_mg, acc_y_mg, acc_z_mg, current_ma, bus_voltage_v);
  } else if (adaptive_learning) {
    finalizeAdaptiveCalibration();
  }

  int adaptive_state = 0;
  float adaptive_prob = 5.0f;
  if (motor_running && adaptive_ready && !in_startup_grace) {
    adaptive_state = evaluateAdaptiveAnomaly(acc_x_mg, acc_y_mg, acc_z_mg, current_ma);
    if (adaptive_state == 2) {
      adaptive_prob = 92.0f;
    } else if (adaptive_state == 1) {
      float vib_dev = computeVibrationDeviation(acc_x_mg, acc_y_mg, acc_z_mg);
      float curr_ratio = (adapt_warn_curr > 1.0f) ? (current_ma / adapt_warn_curr) : 1.0f;
      float vib_ratio = (adapt_warn_vib > 1.0f) ? (vib_dev / adapt_warn_vib) : 1.0f;
      float severity = fmaxf(curr_ratio, vib_ratio);
      adaptive_prob = fminf(85.0f, 45.0f + severity * 25.0f);
    }
  }

  int predicted_state = adaptive_state;
  float predicted_prob = adaptive_prob;

  if (window_ready && interpreter != nullptr && input_tensor != nullptr && output_tensor != nullptr) {
    int k = 0;
    for (int i = 0; i < WINDOW_SIZE; i++) {
      for (int j = 0; j < FEATURE_SIZE; j++) {
        const float v = feature_window[i][j];
        if (input_tensor->type == kTfLiteInt8) {
          int q = (int)roundf(v / input_tensor->params.scale) + input_tensor->params.zero_point;
          if (q < -128) q = -128;
          if (q > 127) q = 127;
          input_tensor->data.int8[k++] = (int8_t)q;
        } else if (input_tensor->type == kTfLiteUInt8) {
          int q = (int)roundf(v / input_tensor->params.scale) + input_tensor->params.zero_point;
          if (q < 0) q = 0;
          if (q > 255) q = 255;
          input_tensor->data.uint8[k++] = (uint8_t)q;
        } else {
          input_tensor->data.f[k++] = v;
        }
      }
    }

    if (interpreter->Invoke() == kTfLiteOk) {
      int classes = output_tensor->bytes;
      if (output_tensor->type == kTfLiteFloat32) classes /= sizeof(float);
      if (output_tensor->type == kTfLiteInt8) classes /= sizeof(int8_t);
      if (output_tensor->type == kTfLiteUInt8) classes /= sizeof(uint8_t);
      if (classes >= 3) {
        float best = dequantizeTensorValue(output_tensor, 0);
        int idx = 0;
        for (int i = 1; i < classes; i++) {
          float p = dequantizeTensorValue(output_tensor, i);
          if (p > best) {
            best = p;
            idx = i;
          }
        }
        // If model output is logits, map roughly into 0..100 via clamped probability-like value.
        float prob_percent = best;
        if (prob_percent <= 1.5f) prob_percent *= 100.0f;
        if (prob_percent < 0.0f) prob_percent = 0.0f;
        if (prob_percent > 100.0f) prob_percent = 100.0f;
        predicted_state = idx;
        predicted_prob = prob_percent;
      }
    }
  }

  // Fuse TinyML model + adaptive self-learned thresholds (take higher severity).
  if (predicted_state < adaptive_state) {
    predicted_state = adaptive_state;
    predicted_prob = adaptive_prob;
  }

  if (in_startup_grace) {
    predicted_state = 0;
    predicted_prob = 0.0f;
  }

  updateAIState(predicted_state, predicted_prob);
}

void TaskAI_Inference(void *pvParameters) {
  (void)pvParameters;
  uint32_t lastLogMs = 0;
  for (;;) {
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    int status = sensor.is_Ready_To_Read();
    if (status == ITDS_enable) {
      sensor.get_acceleration_X(&ax);
      sensor.get_acceleration_Y(&ay);
      sensor.get_acceleration_Z(&az);
    }

    float busv = ina219.getBusVoltage_V();
    float curr = ina219.getCurrent_mA();

    xSemaphoreTake(dataMutex, portMAX_DELAY);
    acc_x_mg = ax;
    acc_y_mg = ay;
    acc_z_mg = az;
    bus_voltage_v = busv;
    current_ma = curr;
    if (motor_running && !system_halted) {
      if (current_baseline_ma < 1.0f) current_baseline_ma = curr;
      current_baseline_ma = 0.96f * current_baseline_ma + 0.04f * curr;
    } else {
      current_baseline_ma = 0.92f * current_baseline_ma + 0.08f * curr;
    }

    pushFeatureWindow(ax, ay, az, curr, busv);
    computeOnDeviceXAI(ax, ay, az, curr, busv);
    runInferenceAndUpdateState();
    applyMotorOutputLocked();
    updateStatusLedLocked();
    xSemaphoreGive(dataMutex);

    if (millis() - lastLogMs > 1000) {
      lastLogMs = millis();
      Serial.printf("AX:%.1f AY:%.1f AZ:%.1f BUS:%.2fV CUR:%.1fmA | adapt:%s state:%d prob:%.1f\n",
                    ax, ay, az, busv, curr,
                    adaptive_learning ? "LEARN" : (adaptive_ready ? "READY" : "WAIT"),
                    current_state, max_prob);
      if (adaptive_ready) {
        Serial.printf("  Thr I warn/dang=%.1f/%.1f | Vib warn/dang=%.1f/%.1f\n",
                      adapt_warn_curr, adapt_dang_curr, adapt_warn_vib, adapt_dang_vib);
      }
    }

    vTaskDelay(pdMS_TO_TICKS(20));
  }
}

void setup() {
  Serial.begin(115200);
  dataMutex = xSemaphoreCreateMutex();

  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZ_PIN, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);

  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZ_PIN, LOW);

  // --- KẾT NỐI VÀO WI-FI LAB & NHẤP NHÁY LED ---
  Serial.print("Connecting to Lab Wi-Fi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN)); // Nhấp nháy liên tục khi chờ kết nối
    delay(250);
    Serial.print(".");
  }

  Serial.println("\nConnected successfully!");
  Serial.print("Local IP Address: ");
  Serial.println(WiFi.localIP());
  digitalWrite(LED_PIN, HIGH); // Sáng báo hiệu hệ thống đã sẵn sàng

  // Khởi tạo các phần cứng I2C (OLED, INA219, v.v.)
  Wire.begin();
  if (display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.display();
  }
  if (!ina219.begin()) {
    Serial.println("[ERROR] INA219 not found. Check wiring.");
  }
  if (sensor.init(ITDS_ADDRESS_I2C_1) == WE_FAIL) {
    Serial.println("[ERROR] WSEN-ITDS init failed. Check wiring/address.");
  } else {
    sensor.ODR = 6; // 200Hz
    sensor.set_High_Performance();
    sensor.set_Full_Scale(3); // +/-16g
  }

  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[ERROR] TFLM schema mismatch.");
  } else {
    interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    if (interpreter->AllocateTensors() != kTfLiteOk) {
      Serial.println("[ERROR] TFLM AllocateTensors failed.");
      interpreter = nullptr;
    } else {
      input_tensor = interpreter->input(0);
      output_tensor = interpreter->output(0);
      Serial.println("[OK] TFLM initialized.");
    }
  }

  // --- THIẾT LẬP CÁC ROUTE WEB SERVER ---
  server.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
    request->send_P(200, "text/html", index_html);
  });

  server.on("/data", HTTP_GET, [](AsyncWebServerRequest *request){
    xSemaphoreTake(dataMutex, portMAX_DELAY);
    String json = "{";
    json += "\"state\":" + String(current_state) + ",";
    json += "\"prob\":" + String(max_prob, 1) + ",";
    json += "\"motor\":" + String(motor_running ? 1 : 0) + ",";
    json += "\"halted\":" + String(system_halted ? 1 : 0) + ",";
    json += "\"ax\":" + String(acc_x_mg, 1) + ",";
    json += "\"ay\":" + String(acc_y_mg, 1) + ",";
    json += "\"az\":" + String(acc_z_mg, 1) + ",";
    json += "\"bus_v\":" + String(bus_voltage_v, 2) + ",";
    json += "\"cur_ma\":" + String(current_ma, 2) + ",";
    json += "\"xai_x\":" + String(xai_contrib_x, 1) + ",";
    json += "\"xai_y\":" + String(xai_contrib_y, 1) + ",";
    json += "\"xai_z\":" + String(xai_contrib_z, 1) + ",";
    json += "\"xai_i\":" + String(xai_contrib_current, 1) + ",";
    json += "\"xai_v\":" + String(xai_contrib_voltage, 1) + ",";
    json += "\"adapt_learning\":" + String(adaptive_learning ? 1 : 0) + ",";
    json += "\"adapt_ready\":" + String(adaptive_ready ? 1 : 0) + ",";
    json += "\"thr_warn_i\":" + String(adapt_warn_curr, 1) + ",";
    json += "\"thr_dang_i\":" + String(adapt_dang_curr, 1) + ",";
    json += "\"thr_warn_v\":" + String(adapt_warn_vib, 1) + ",";
    json += "\"thr_dang_v\":" + String(adapt_dang_vib, 1);
    json += "}";
    xSemaphoreGive(dataMutex);
    request->send(200, "application/json", json);
  });

  server.on("/start", HTTP_POST, [](AsyncWebServerRequest *request){
    xSemaphoreTake(dataMutex, portMAX_DELAY);
    if (!system_halted) {
      startMotorLocked();
    }
    xSemaphoreGive(dataMutex);
    request->send(200, "text/plain", "OK");
  });

  server.on("/stop", HTTP_POST, [](AsyncWebServerRequest *request){
    xSemaphoreTake(dataMutex, portMAX_DELAY);
    stopMotorLocked();
    xSemaphoreGive(dataMutex);
    request->send(200, "text/plain", "OK");
  });

  server.on("/reset", HTTP_POST, [](AsyncWebServerRequest *request){
    xSemaphoreTake(dataMutex, portMAX_DELAY);
    system_halted = false; motor_running = false; current_state = 0; max_prob = 0; warning_count = 0;
    startup_boost_active = false;
    current_baseline_ma = 0.0f;
    adaptive_learning = false;
    adaptive_ready = false;
    digitalWrite(BUZ_PIN, LOW);
    applyMotorOutputLocked();
    xSemaphoreGive(dataMutex);
    request->send(200, "text/plain", "OK");
  });

  server.begin();
  
  // Ghim tác vụ AI (Inference + sensor sampling) vào Core 1
  xTaskCreatePinnedToCore(TaskAI_Inference, "TaskAI", 12288, NULL, 1, NULL, 1);
}

void loop() {
  // --- ĐIỀU KHIỂN ĐỒNG THỜI QUA SERIAL (PuTTY) ---
  if (Serial.available()) {
    char cmd = Serial.read();
    xSemaphoreTake(dataMutex, portMAX_DELAY);
    if (cmd == 'a' && !system_halted) {
      startMotorLocked();
      Serial.println(">> Motor Started via Serial");
    } 
    else if (cmd == 'q') {
      stopMotorLocked();
      Serial.println(">> Motor Stopped via Serial");
    } 
    else if (cmd == 'r') {
      system_halted = false; motor_running = false; current_state = 0; max_prob = 0; warning_count = 0;
      startup_boost_active = false;
      current_baseline_ma = 0.0f;
      adaptive_learning = false;
      adaptive_ready = false;
      digitalWrite(BUZ_PIN, LOW);
      applyMotorOutputLocked();
      Serial.println(">> System Reset via Serial");
    }
    xSemaphoreGive(dataMutex);
  }

  // --- CẬP NHẬT MÀN HÌNH OLED 3 DÒNG THEO TIÊU CHUẨN ---
  xSemaphoreTake(dataMutex, portMAX_DELAY);
  int st = current_state;
  float pr = max_prob;
  bool isRunning = motor_running;
  bool isHalted = system_halted;
  xSemaphoreGive(dataMutex);

  display.clearDisplay();
  display.setCursor(0, 0);
  
  // Dòng 1: IP của mạng phòng Lab
  display.print("IP: ");
  display.println(WiFi.localIP());

  // Dòng 2: Trạng thái cơ cấu chấp hành (Động cơ)
  display.print("Motor: ");
  if (isHalted) display.println("HALTED");
  else if (isRunning) display.println("RUNNING");
  else display.println("STOPPED");

  // Dòng 3: Kết quả phân tích biên cục bộ từ Edge AI
  display.print("AI: ");
  if (st == 0) display.print("NORMAL");
  else if (st == 1) display.print("WARNING");
  else display.print("DANGER");
  display.printf(" (%d%%)", (int)pr);
  
  display.display();
  delay(300); // Giảm tải xử lý bus I2C
}