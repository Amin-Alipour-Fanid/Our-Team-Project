# Real-time Iranian Sign Language Recognition (ISLR101)

پروژه تشخیص زبان اشاره ایرانی به صورت **لحظه‌ای** با استفاده از وب‌کم.

مدل اصلی با معماری **BiGRU** روی دیتاست ISLR101 (۱۰۱ کلاس) آموزش دیده است.  
در حالت real-time از **MediaPipe** برای استخراج نقاط کلیدی بدن و دست‌ها استفاده می‌شود و سپس با کالیبراسیون، ورودی به فضای مدل آموزش‌دیده نگاشت می‌شود.

---

## ویژگی‌ها

- تشخیص real-time زبان اشاره ایرانی
- پشتیبانی از ۱۰۱ کلاس
- کالیبراسیون خودکار برای هماهنگی بین MediaPipe و داده‌های آموزش
- نمایش نتیجه به فارسی روی تصویر
- امکان کنترل دستی شروع/پایان اشاره (کلید `m` و `SPACE`)

---

## نصب

```bash
# کلون کردن پروژه
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# ساخت محیط مجازی (پیشنهادی)
python -m venv venv

# ویندوز:
venv\Scripts\activate

# لینوکس/مک:
source venv/bin/activate

# نصب پکیج‌ها
pip install -r requirements.txt
```

---

## ساختار پروژه

```text
├── model.py                  # معماری BiGRU
├── preprocessing.py          # پیش‌پردازش و DataLoader
├── keypoint_extractor.py     # استخراج نقاط کلیدی با MediaPipe
├── calibration.py            # کالیبراسیون فضای ویژگی
├── input_spec.py             # مشخصات ورودی مدل
├── train.py                  # آموزش مدل
├── evaluate.py               # ارزیابی
├── test.py                   # تست ساده
├── realtime_inference.py     # موتور استنتاج real-time
├── realtime_demo.py          # دموی زنده با وب‌کم
├── audit_phase0.py           # بررسی داده‌ها
└── fix_t5_phase1.py          # تست MediaPipe Tasks API
```

---

## نحوه استفاده (Real-time)

### ۱. کالیبراسیون (فقط یک بار)

```bash
python calibration.py
```

این اسکریپت فایل `checkpoints/calibration.json` را می‌سازد.

### ۲. تست آفلاین (اختیاری)

```bash
python realtime_inference.py
```

### ۳. اجرای دموی زنده

```bash
python realtime_demo.py
```

**نکات مهم هنگام استفاده:**

- دست‌ها را پایین بیاورید → اشاره را انجام دهید → دوباره دست‌ها را پایین بیاورید.
- اگر تشخیص شروع/پایان اشاره خوب کار نمی‌کند، کلید **`m`** را بزنید تا به حالت دستی بروید و با **`SPACE`** شروع و پایان هر اشاره را مشخص کنید.
- اگر تصویر وب‌کم شما آینه‌ای است، داخل فایل `realtime_demo.py` این خط را تغییر دهید:

```python
FRAME_IS_MIRRORED = True
```

---

## پیش‌نیازها برای Real-time

- وب‌کم
- مدل‌های MediaPipe (اولین اجرا به صورت خودکار دانلود می‌شوند)
- فایل‌های checkpoint مدل آموزش‌دیده (باید در پوشه `checkpoints/` قرار گیرند)

---

## مجوز

این پروژه برای اهداف آموزشی و تحقیقاتی منتشر شده است.
