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
### 🧠 Распознаваемые объекты

Модель `YOLOv8n` обучена на 80+ объектах на COCO-датасете и может определять:

- Людей и животных (человек, собака, кошка)
- Транспорт (автомобиль, автобус, велосипед)
- Объекты (телефон, бутылка, ноутбук, пицца и др.)

---

### 🖐️ Поддерживаемые жесты

| Жест       | ASCII-графика | Распознавание        | Цвет подсветки |
|------------|----------------|-----------------------|----------------|
| 👍         | `\o/`          | Thumbs Up             | Тёмно-жёлтый   |
| ✊         | `[###]`        | Сжатый кулак (Fist)   | Красный        |
| 🖐️         |      | Открытая ладонь       | Зелёный        |
| ☝️         |          | Один поднятый палец   | Тёмно-жёлтый   |

---

### 🧪 Управление

| Кнопка | Действие                |
|--------|-------------------------|
| `o`    | Включить YOLO-режим     |
| `h`    | Включить жесты рук      |
| `q`    | Выход из программы      |

---

### ⚙️ Установка
```bash
git clone https://github.com/NURJAKS/VisionCore.git
cd VisionCore
python3 -m venv venv
source venv/bin/activate
```
Скопируйте и вставьте следующую команду в терминал:

```bash
pip install ultralytics mediapipe opencv-python
```
🚀 Запуск
python main.py

💡 Используемые технологии

🟢 YOLOv8 — детекция объектов

🟠 MediaPipe — распознавание рук

🔵 OpenCV — видеопоток и визуализация

👤 Автор
Nurbek Abildaev
