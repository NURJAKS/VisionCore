<h1 align="center">🧠 VisionCore</h1>
<p align="center">Real-time object and hand gesture recognition using YOLOv8 and MediaPipe</p>

<p align="center">
  <img src="https://img.shields.io/badge/OpenCV-vision-blue" />
  <img src="https://img.shields.io/badge/MediaPipe-hands-orange" />
  <img src="https://img.shields.io/badge/YOLOv8-detection-green" />
</p>

---

### 🎯 Описание

**VisionCore** — это мощный AI-инструмент, который сочетает в себе:
- Распознавание объектов через **YOLOv8**
- Детекцию жестов руки с помощью **MediaPipe**
- Переключение режимов в реальном времени

---

###🖐️ Поддерживаемые жесты
\o/ — Thumbs Up — тёмно-жёлтый

[###] — Fist — красный

[|||||] — Open Palm — зелёный

|__ — One Finger — тёмно-жёлтый

---

### 🧪 Управление

| Кнопка | Действие                |
|--------|-------------------------|
| `o`    | Включить YOLO-режим     |
| `h`    | Включить жесты рук      |
| `q`    | Выход из программы      |

---

### ⚙️ Установка
git clone https://github.com/NURJAKS/VisionCore.git

cd VisionCore

python3 -m venv venv
source venv/bin/activate

pip install opencv-python mediapipe ultralytics

🚀 Запуск
python main.py
💡 Используемые технологии

🟢 YOLOv8 — детекция объектов

🟠 MediaPipe — распознавание рук

🔵 OpenCV — видеопоток и визуализация

👤 Автор
Nurbek Abildaev
