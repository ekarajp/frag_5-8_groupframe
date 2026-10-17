# Component IDA / Fragility — RC Grouped-frame ชั้น 5–8

โค้ดสำหรับ **2,000 โมเดล: ชั้นละ 500** ใช้คลื่น CMS เดิม 23 คู่ เลือก 3–4 คู่ต่ออาคาร ⇒ **6–8 component IDA curves** แยก IM, scaling และ capacity ของ X/Y มี dashboard Start / Stop / Workers และความคืบหน้ารายชั้น/รวม คิวเติมทันทีเมื่อ worker ว่าง

**Linux x86_64 และ Windows x64:** ติดตั้ง Python 3.10 ตามด้านล่าง **macOS รวม Apple Silicon:** ใช้ Docker `linux/amd64` เพื่อคง OpenSees รุ่นเดียวกับงานเดิม (การจำลอง x86 บน ARM อาจช้ากว่า native) ไม่รองรับทุก OS/architecture โดยอัตโนมัติ

## ติดตั้งจากโค้ด: Linux / Windows

```bash
git clone https://github.com/ekarajp/frag_5-8_groupframe.git
cd frag_5-8_groupframe
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/prepare_data.py
python tools/doctor.py --solver-import
python app/verify.py
python run.py
```

Windows ใช้ `py -3.10 -m venv .venv` และ `.venv\Scripts\activate` แทนสองคำสั่งแรกของ Python จากนั้นใช้ `python` เหมือนกัน หรือเปิด `START_FRAGILITY.bat` เมื่อเตรียม environment/ข้อมูลครบแล้ว

Linux Debian/Ubuntu อาจต้องติดตั้ง `python3.10-venv libgomp1 libglu1-mesa` ก่อน ชื่อแพ็กเกจ Python ขึ้นกับ distribution คำสั่ง `doctor --solver-import` ทดสอบ import/version เท่านั้น **ไม่มี analyze()**

เปิด **http://127.0.0.1:8788** แล้วเลือก workers และกด Start เอง การเปิด dashboard **ไม่เริ่ม solver** สำหรับเครื่อง Linux ไม่มีจอ ใช้ `python run.py --no-browser` และ SSH tunnel:

```bash
ssh -L 8788:127.0.0.1:8788 user@your-server
```

## macOS / Docker Desktop / Linux Docker

ใช้ Docker Engine 23+ กับ Compose v2 (BuildKit) — build ดาวน์โหลดข้อมูลชั้น 5 (~643 MB) จาก GitHub Release เอง ตรวจ SHA-256 ทุกไฟล์ ไม่ต้องมี Python บนเครื่อง host อีกต่อไป:

```bash
docker compose up --build
```

เปิด http://127.0.0.1:8788 ผลอยู่ใน `state/` บนเครื่อง host และ cache ใน `runtime_storage/` เปิดเฉพาะ loopback ไม่ต้องเปิดพอร์ต dashboard สู่ Internet ก่อน `docker compose down` ให้กด Stop แล้วรอ active=0 เพื่อเก็บจุดปัจจุบันครบ ห้ามเปิดสอง controller ต่อ state เดียวกัน

เครื่อง Linux server ไม่มีจอ หรือเข้าถึงผ่าน LAN: ดู [docs/DEPLOY_UBUNTU.md](docs/DEPLOY_UBUNTU.md) — ตั้ง `FRAGILITY_BIND`/`FRAGILITY_ALLOWED_HOSTS` ใน `.env` สร้างโฟลเดอร์ `state/` `runtime_storage/` ให้เป็นของ UID 1000 ก่อน `up` ครั้งแรก แล้วเปิด `http://<server-ip>:8788`

## ไฟล์และข้อมูลที่ได้รับ

- Git: application, source snapshots แยกชั้น 5 กับ 6–8, โมเดลเดิม 2,000 ไฟล์, recipes, catalog order, manifest, คลื่น 46 components และการตรวจแบบไม่วิเคราะห์
- GitHub Release `research-data-v1`: directional histories ชั้น 5 เดิม 192 observations และ metadata จำนวน 384 ไฟล์ (~643 MB) เพื่อใช้ต่อเมื่อ identity/period/candidate/hash ตรงกัน `tools/prepare_data.py` ดาวน์โหลดและตรวจ SHA-256 ทุกไฟล์ก่อนใช้ หรือใช้ `--archive /path/to/floor5-legacy-seeds-v1.zip` แบบ offline
- ไม่ใส่ Windows `python.exe` ลง Git แต่ติดตั้ง native dependencies ของเครื่องปลายทางจาก `requirements.txt` ซึ่งตรึง OpenSeesPy และ native Windows/Linux engine **3.5.1.3** ตามต้นฉบับ
- แพ็กเกจ Windows เดิมและรายงาน/ประวัติเปรียบเทียบขนาดใหญ่ยังคงอยู่ในเครื่องวิจัย ไม่ถูกแก้หรือลบ รายงานเก่าบางลิงก์ใน metadata จึงไม่ได้แนบใน Git ชุดนี้ การรันใหม่สร้างรายงานของตัวเองครบ

`bundle_manifest.json` เป็น manifest รุ่น source distribution ใหม่ แยกจาก `distribution/original_windows_bundle_manifest.json` ที่เก็บหลักฐานแพ็กเกจเดิม การปรับระบบปฏิบัติการมีบันทึกใน `distribution/crossplatform_declaration.json` ไม่มีการเปลี่ยน source กลศาสตร์ โมเดล คลื่น หรือ threshold เพื่อให้ผ่าน

## วิธีคำนวณและขอบเขต

IM = Sa ของ **component ที่ T1 ทิศนั้น, damping 5%**, own SF = own IM / own unscaled Sa ไม่รวม max X/Y เป็นเส้นเดียว IO/LS/CP = .01/.02/.04, hunt + log bisection, bracket ≤5% ค่า guard/failure เก็บ needs_review ไม่แทนเป็น CP และไม่ retry signed failure เดิมอัตโนมัติ Fragility/bootstrap เก็บ parent-pair grouping ไม่อ้างว่า components จากเหตุการณ์เดียวกันเป็นเหตุการณ์อิสระ

ชั้น 5 ใช้ mesh16 ล่าสุด ส่วน 6–8 ใช้โมเดลส่งมอบเดิมทั้งไฟล์ รายละเอียดวิทยาศาสตร์/ผลที่เก็บอยู่ใน [คู่มือเดิม](README_TH.md) และ [วิธีนำผลกลับ](RETURN_RESULTS_TH.md) โดยคำสั่งติดตั้งในหน้านี้ใช้แทนคำสั่ง bundled Windows runtime ในคู่มือเดิม

เมื่อย้ายเครื่อง/OS: Stop → active=0 → ปิด controller → คัดลอก **state ทั้งโฟลเดอร์** พร้อม source/assets รุ่นเดิม ตรวจ environment/version และ hashes ก่อน Resume ความต่างเชิง floating point ข้ามแพลตฟอร์มอาจเกิดได้ จึงบันทึก Python/platform/package versions แยกไว้ ไม่อ้างผลเท่ากันทุกบิตข้าม OS

## ตรวจโดยไม่รัน population

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python tests/portability_no_analysis.py
python tests/seed_probe.py
python app/verify.py
```

Queue tests ใช้ fake worker, verifier โหลด X/Y construction commands 4,000 ครั้งโดยไม่เรียก OpenSees วิเคราะห์ GitHub Actions ตรวจ Linux/Windows ส่วน macOS ใช้ Linux Docker image การผ่านชุดตรวจนี้ไม่ใช่การรับรอง fragility ครบ 2,000 อาคาร ต้องรัน campaign ที่เครื่องปลายทางก่อน
