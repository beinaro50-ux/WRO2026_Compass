# WRO2026_Compass
# Team Compass — WRO 2026 Future Engineers: Self-Driving Cars Challenge

**Country:** Cambodia
**Team Members:** Be Naro, Yeong Vechakasy Sothon, Art Oudom
**Coach:** Taing Liv Min

**GitHub Repository:** `[ TBD — add before the 3-weeks-before-competition submission deadline ]`
**Open Challenge Video:** `[ TBD ]`
**Obstacle Challenge Video:** `[ TBD ]`

---

## Table of Contents
1. [Mobility and Mechanical Design](#1-mobility-and-mechanical-design)
2. [Power and Sensor Architecture](#2-power-and-sensor-architecture)
3. [Software Architecture and Obstacle Strategy](#3-software-architecture-and-obstacle-strategy)
4. [Systems Thinking and Engineering Decisions](#4-systems-thinking-and-engineering-decisions)
5. [Reproducibility and Repository Structure](#5-reproducibility-and-repository-structure)
6. [Photos and Video Demonstrations](#6-photos-and-video-demonstrations)

---

## 1. Mobility and Mechanical Design

### 1.1 Chassis and Drive System
Our chassis is fully **3D-printed in PLA filament**, designed to fit within the 300 × 200 × 300 mm vehicle envelope required by WRO rule 11.1.

The drivetrain is **rear-wheel drive (RWD)**, powered by a single **JGA25-370 DC gear motor** (12V, rated 1360 RPM), driving the rear axle through a two-stage gear reduction:
- A 14 mm pinion gear on the motor output meshes with a 25 mm gear on the driven axle
- This gives an approximate **1.79:1 reduction**

This gearing trades top speed for extra torque and steadier low-speed control, which matters for precise lane-following and parking maneuvers.

### 1.2 Steering Mechanism
Steering is handled by a single **MG996R servo** connected to a steering linkage on the front axle, satisfying WRO rule 11.3 (one steering actuator, front/rear/4WD drive — not a differential/skid-steer base).

| Parameter | Value |
|---|---|
| Length from servo to wheel connector | 75 mm |
| Maximum steering angle | 30° |
| Radius | 75 mm |
| Diameter | 150 mm |
| Turning length (AB) | 39.25 mm |

### 1.3 Diagrams / CAD Views
Robot 
<img width="749" height="440" alt="image" src="https://github.com/user-attachments/assets/6398ca0d-57a4-4c79-a000-61214774ea21" />

Front side
<img width="819" height="472" alt="image" src="https://github.com/user-attachments/assets/8040a737-4df6-4a29-babd-0f14af116006" />

Back side
<img width="787" height="490" alt="image" src="https://github.com/user-attachments/assets/b440b215-5d06-489a-8084-25cdc805ef57" />

Bottom 
<img width="743" height="519" alt="image" src="https://github.com/user-attachments/assets/0697602e-26a2-4d50-a373-a280738f76a4" />






### 1.4 Testing and Iteration
We went through **three major chassis redesigns** before arriving at our current layout:

- **Versions 1 & 2:** Placed the RPLiDAR A2M12 at the rear of the vehicle. Limited internal space made it difficult to route wiring and mount the ESP32, L298N driver, and battery cleanly.
- **Version 3 (current):** Relocated the LiDAR to the front of the vehicle and moved the ESP32, motor driver, and battery to the rear. This freed up space around the LiDAR's 360° scanning field and simplified our wiring harness.

We also **raised the camera's mounting height** after early testing showed our original (lower) mounting position caused the camera to miss or misjudge the distance to obstacle pillars. Raising the mount improved our ability to keep pillars clearly framed for detection.

---

## 2. Power and Sensor Architecture

### 2.1 Power System
The vehicle uses a single **12V 3A battery pack** as the primary power source, feeding the ESP32, the DC motor (via the L298N driver), and the steering servo. The servo line is stepped down from 12V through a regulator/transformer before reaching the MG996R (rated 4.8–7.2V).

The **Raspberry Pi 4 is powered independently** from a separate portable USB power bank, isolating its 5V logic supply from motor and servo electrical noise on the 12V line.

### 2.2 Power Budget

| Component | Voltage | Est. Current | Source |
|---|---|---|---|
| Raspberry Pi 4 | 5V (power bank) | ~3A max | Official RPi4 PSU rating |
| ESP32 | 5V (onboard reg.) | ~0.3–0.5A typical | Datasheet typical |
| RPLiDAR A2M12 | 5V | ~0.2–0.3A (core+motor) | Manufacturer spec |
| JGA25-370 DC Motor | 12V | ~0.3A run / ~3A stall | Typical for motor class |
| MG996R Servo | 9V | ~0.5–0.9A / ~2.5A stall | Datasheet typical |

> **Note:** These are datasheet/spec estimates, not yet measured on this specific build. Actual draw should be measured with a multimeter or USB power meter (especially motor and servo under load/stall) before finalizing wiring gauge and battery sizing.

### 2.3 Sensor Suite
- **RPLiDAR A2M12** (360°, 12 m range) — mounted front-facing, providing wall-distance and obstacle-range data used for LiDAR-based wall following.
- **USB webcam (8 MP)** — mounted toward the back of the chassis on a camera stand. Used for pillar color/class detection and lane-position estimation via our trained YOLO model.

### 2.4 Wiring Diagram
The vehicle's electrical system is split into four subsystems on two isolated power domains: a 5V logic/compute domain (Raspberry Pi 4 + sensors) and a 12V power domain (drive motor + servo), bridged by the ESP32 acting as the real-time actuator controller.
Subsystem 1 — Central Control & Vision (5V)
- A dedicated 5V powerbank supplies the Raspberry Pi 4 via USB 3.0 (rated ~3A max draw).
- The USB 2.0 8MP PC camera connects to the Pi 4 via USB 3.0.
- The RPLiDAR A2M12 (5V, ~0.2–0.3A) connects to the Pi 4 via USB.
- This subsystem is intentionally powered independently from the 12V motor/servo domain, isolating the Pi's logic supply from motor electrical noise.
Subsystem 2 — Actuator Control (ESP32). 
- The Raspberry Pi 4 communicates with the ESP32 over a serial link (Tx/Rx), sending high-level driving commands.
- The ESP32 runs on its onboard 5V regulator (~0.3–0.5A).
- The ESP32 outputs: 
  **PWM + H-bridge control signals to the L298N motor driver (Subsystem 3)**
  **PWM signal to the MG996R servo (Subsystem 4)**
  **Shared GND to both subsystems**
  Subsystem 3 — L298N DC Motor Driver (12V high-power)
- A 12V 3A battery feeds the L298N driver board directly ("12V Battery Input").
- The JGA25-370 DC motor connects to the L298N's Motor A output terminals.
- The ESP32 sends a PWM signal (IN1) to the L298N to control motor speed/direction.
- The L298N also exposes a 12V output tap, which is routed onward to Subsystem 4's power converter.
  Subsystem 4 — Servo Subsystem
- The L298N's 12V output feeds a power converter (12V in → 9V out), which steps the battery voltage down to a safe level for the MG996R servo (rated 4.8–7.2V nominal, but run here at ~9V converted power per your table).
- The servo receives PWM control from the ESP32 and shares GND across the system.
- Current draw reference from your spec table: ~0.33A running at 12V, ~0.5–0.9A running at 9V, up to ~2.5A stall.


### 2.5 Calibration
*[ Describe sensor calibration — e.g., camera color thresholds for red/green pillar detection, LiDAR mounting angle offset — once finalized ]*

---

## 3. Software Architecture and Obstacle Strategy

### 3.1 System Architecture
Our software runs on the **Raspberry Pi 4** and is organized into cooperating modules:

- **LiDAR reader** — background thread continuously reading the RPLiDAR A2M12
- **Vision worker** — camera capture plus YOLO model inference
- **State machine** — fuses LiDAR wall-following with camera-based pillar and lane detection to decide speed and steering
- **Serial link** — sends the resulting commands to the ESP32, which drives the DC motor (via L298N) and steering servo in real time

**Flowchart**
<img width="472" height="362" alt="image" src="https://github.com/user-attachments/assets/4844aa66-f046-4ead-884d-3ca2eae206d3" />


### 3.2 Lane Following Strategy
Our primary lane-following signal comes from the **RPLiDAR A2M12**: we calculate the vehicle's distance to the left and right walls from the 360° scan and use a **PID controller** to steer toward the section centerline.

Camera-based lane detection from our YOLO model is layered on top as a **secondary correction signal**, since LiDAR is more robust to lighting changes on the track than a camera alone.

### 3.3 Obstacle (Traffic Sign) Strategy
Our team designed the vehicle's mechanical layout in **Solid Edge** and trained our pillar/lane detection model using **YOLO**. The model detects red and green traffic pillars in the camera frame and classifies them by color.

Per WRO rule 9.19:
- When a **red pillar** is detected ahead, the vehicle steers to keep the pillar on its **right** side.
- When a **green pillar** is detected, the vehicle steers to keep the pillar on its **left** side.


**To be completed before submission:**
- Final training image count and train/validation split
- Key metrics (e.g., mAP, precision/recall)
- PID tuning notes for the wall-following controller, based on physical track testing


## 4. Systems Thinking and Engineering Decisions

### 4.1 Subsystem Interaction Map
<img width="491" height="198" alt="image" src="https://github.com/user-attachments/assets/c461ad57-4f66-46c3-a32a-6b8bdae002c9" />

That maps how your four subsystems depend on each other: Power feeds all three others directly (amber arrows), Sensors feed raw LiDAR + camera data into Software, and Software sends the final steering/motor commands down to Mobility.

This is the kind of diagram Appendix C Criterion 4 ("Systems Thinking") specifically rewards — it shows you understand the robot as an interconnected system rather than just a parts list. A couple of notes for when you drop this into the README:

- If you want to be even more precise, you could split the single "Power" arrow into three separately-labeled ones (5V line to Pi/sensors, 12V line to motor, 9V converted line to servo) — that would tie directly back into your power budget table and reinforce the two-domain design you documented.
- This diagram pairs well with the ground-domain risk we discussed earlier — you could add a small dashed line or note showing "no shared ground (current risk)" between the Power and Mobility boxes if you want the diagram itself to visually flag that gap, not just the prose.


### 4.2 Key Design Tradeoffs

**Decision 1 — Sensor placement vs. available chassis space:**
Our first two chassis revisions placed the RPLiDAR at the rear of the vehicle, but tight internal space made wiring and component mounting difficult. We chose to redesign (third iteration) and move the LiDAR to the front, trading some rear layout simplicity for a cleaner wiring harness and an unobstructed 360° scan field.

**Decision 2 — ONNX vs. PyTorch for on-board inference:**
Our trained model was exported in both PyTorch (`.pt`) and ONNX (`.onnx`) formats. We chose to deploy the **ONNX** version on the Raspberry Pi 4, since ONNX Runtime is significantly faster and lighter-weight for inference on limited onboard compute than running the full PyTorch framework — this matters for keeping our control loop running at a consistent rate during a live round.

### 4.3 Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Serial link between RPi4 and ESP32 drops mid-run | ESP32 firmware includes a watchdog timer that force-stops the motor if no command is received within 300ms |
| Servo over-voltage (MG996R rated 4.8–7.2V, battery is 12V) | Servo power is stepped down through a separate regulator before reaching the servo (exact output voltage to be confirmed and documented) |
| Camera mounted too low to reliably detect pillars | Identified during testing; camera mount height was raised in our third chassis revision |
| Limited internal chassis space causing wiring/component conflicts | Redesigned chassis three times, relocating LiDAR to the front and ESP32/motor driver/battery to the rear |

### 4.4 Version History
- **Version 1:** Initial chassis with LiDAR mounted at the rear — space too tight for clean wiring.
- **Version 2:** Adjusted internal layout, LiDAR remained at the rear — still constrained; camera mounted too low for reliable pillar detection.
- **Version 3 (current):** LiDAR relocated to the front; ESP32, L298N, and battery relocated to the rear; camera mount height raised.

We are going to fix some of this robot machinal due to it contains some stress from steering wheel and the battery do not have enough power to 

---

## 5. Reproducibility and Repository Structure

### 5.1 Planned Repository Layout

```
/src              — vehicle control source code
                    (main.py, config.py, lidar_module.py,
                     vision_module.py, state_machine.py, serial_link.py)
/esp32_firmware   — ESP32 motor/servo controller firmware
/models           — trained YOLO model files (ONNX export used on-vehicle)
/cad              — 3D-printable chassis files
/schematics       — wiring diagrams, power budget
/media            — vehicle photos (6 views) and team photo
```

> **Status:** Repository is not yet public. Per WRO rules, we will follow the required commit timing (first commit ≥ 2 months before competition containing at least 1/5 of final code, second ≥ 1 month before, third ≥ 2 weeks before) and add the public URL above before the submission deadline (3 weeks before competition).

### 5.2 Build Instructions Summary
**Software section** 
We use a pretrained YOLO model (not custom-trained on our own pillar dataset) rather than collecting and labeling our own training set. Our own engineering work here was in configuring the model's confidence threshold — tuning the minimum detection confidence YOLO needs before we trust a detection, to balance:

Missed detections (threshold too high → pillar not seen in time to react)
False positives (threshold too low → reacting to noise or irrelevant objects)

Why we chose this approach over training our own model:
Training a custom YOLO model requires collecting and labeling a large dataset of red/green pillar images across different lighting conditions, distances, and angles — a significant time investment. Using a pretrained model with tuned thresholds let us get a working detection pipeline running quickly at the tradeoff of relying on YOLO's general-purpose object classes rather than a model purpose-built to recognize WRO pillars specifically.

To be completed before submission:

The exact confidence threshold value we settled on, and the reasoning/testing behind it (e.g., "we tested 0.5, 0.6, 0.7 and found 0.6 gave the best balance of X vs Y")
How color (red vs. green) is actually determined once YOLO flags a detection — e.g., pixel-color sampling within the bounding box (this connects to your calibration section, which is also still pending)
Any measured detection accuracy/reliability from track testing

**Hardware Section**
The mechanical design phase took approximately two weeks, followed by roughly nine days of physical assembly.

A key challenge emerged during this process: our initial chassis design was built to accommodate only three hardware components. Partway through assembly, some of those original components broke and had to be replaced with larger alternatives — which no longer fit within the space our first design had allocated. This forced us to revisit our internal layout and rework the chassis to accommodate the new component sizes, rather than simply swapping parts in place.

This experience directly informed our decision to design for extra internal clearance in later chassis revisions (see Version History, Section 4.4), rather than sizing the chassis tightly around the exact components on hand — a lesson that also shaped our approach to component selection going forward, favoring some margin for hardware substitutions over a minimal-footprint design.

## 6. Photos and Video Demonstrations

### 6.1 Required Photos
- Team photo
  <img width="1280" height="960" alt="photo_2026-08-28_14-54-29" src="https://github.com/user-attachments/assets/57bb75e5-9713-48f3-88cc-5026171fd9ac" />
  
- Front side
  <img width="4032" height="3024" alt="IMG_8092" src="https://github.com/user-attachments/assets/f6cc348d-d787-4123-b148-7f806f56c99d" />
- Back Side
  <img width="4032" height="3024" alt="IMG_8101" src="https://github.com/user-attachments/assets/61f19d81-04a1-4880-8a73-8d39bb74dccf" />
- Above
  <img width="4032" height="3024" alt="IMG_8094" src="https://github.com/user-attachments/assets/4c2090c6-e494-4e5e-8571-a6fa91a6c560" />
- Bottom
  <img width="4032" height="3024" alt="IMG_8102" src="https://github.com/user-attachments/assets/422b9ee8-ad6a-40bf-b2ab-ab48b28f5f5b" />
- Left Side
  <img width="4032" height="3024" alt="IMG_8098" src="https://github.com/user-attachments/assets/12210211-723f-4efe-8ea4-279dc5a33cdf" />
- Right Side
  <img width="4032" height="3024" alt="IMG_8100" src="https://github.com/user-attachments/assets/610e2716-837f-4a71-b641-28d5d480b344" />


### 6.2 Video Demonstrations
Per WRO documentation rules, one YouTube video (public or unlisted/link-accessible) is required for **each** challenge, with a driving demonstration segment at least 30 seconds long.

- **Open Challenge video:** `[ TBD ]`
- **Obstacle Challenge video:** `[ TBD ]`

---

*This documentation follows the WRO 2026 Future Engineers General Rules, Section 7 (Engineer's Documentation on GitHub) and Appendix C (Engineering Journal Evaluation).*
