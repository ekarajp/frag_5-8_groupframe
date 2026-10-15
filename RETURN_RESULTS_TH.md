# ส่งโปรแกรมไปเครื่องอื่น และส่งผลกลับ

## เริ่มรันบนเครื่องปลายทาง

1. แตก ZIP ให้ครบ จะได้โฟลเดอร์ `fragility-runner-5-8` อย่ารันจากภายในหน้าต่าง ZIP
2. เปิด `VERIFY_PACKAGE.bat` เพื่อตรวจ hash และโมเดล 2,000 อาคาร / X,Y 4,000 definitions แบบไม่วิเคราะห์
3. เปิด `START_FRAGILITY.bat` แล้วเข้า dashboard ที่ `http://127.0.0.1:8788`
4. เลือก workers ให้เหมาะกับเครื่องแล้วกด Start / Resume การเปิดโปรแกรมอย่างเดียวไม่เริ่ม solver

มี runtime, source แยกตามชั้น, CMS เดิม 23 คู่, โมเดลชั้น5–8 ชั้นละ500 และข้อมูลชั้น5ที่นำมาใช้ต่อได้อยู่ภายในทั้งหมด ชั้น5ใช้ mesh16 ล่าสุด ผล directional เดิม192จุดถูกเก็บไว้ ในจำนวนนี้72จุดจาก3อาคารตรงกับ catalog500 ก่อนเลือก family ใช้ต่อเฉพาะ model/candidate/component/GM/T1/hash ตรงกัน ไม่ผสมผล mesh8

## ก่อนคัดลอกผลกลับ

1. ตรวจจำนวน **certified** และรายการ **needs_review** ด้วย จำนวน processed=2,000 อย่างเดียวไม่ได้หมายความว่าได้ fragility ครบทุกอาคาร
2. กด **Stop safely** และรอ native workers active=0
3. กด **Export research inventory** แล้วรอ inventory/export complete เพื่อให้ได้รายงานรวมและรายการ hash ล่าสุด หากผลยังไม่ครบ รายงานต้องแสดง incomplete ตามจริง
4. ปิดหน้าต่างโปรแกรม/หน้าต่าง Python ที่เปิดจาก START_FRAGILITY.bat การปิดแท็บเบราว์เซอร์อย่างเดียวไม่ใช่การปิด server
5. เปิด **VERIFY_RESULTS.bat** หลังปิดโปรแกรมแล้ว เก็บผลตรวจใน `validation/results_audit.json` ด้วย ถ้ามี errors ให้ส่งกลับพร้อมข้อมูลทั้งหมด อย่าลบหรือแก้ผลให้ดูเหมือนผ่าน
6. คัดลอกเมื่อไม่มีโปรแกรมหรือ worker เขียนไฟล์แล้วเท่านั้น เก็บไฟล์ SQLite ที่มีนามสกุล `-wal` / `-shm` ไปด้วยถ้ายังมีอยู่

## วิธีที่ง่ายและครบที่สุด

คัดลอก **โฟลเดอร์ `fragility-runner-5-8` ทั้งโฟลเดอร์หลังรัน** กลับมา จะได้ทั้ง input เดิม โปรแกรม และผลทุกอย่างพร้อมตรวจย้อนกลับหรือ Resume ภายหลัง

## ถ้าต้องการลดขนาดที่ส่งกลับ

เมื่อโปรแกรม โมเดล คลื่น และ config ไม่ได้เปลี่ยนจาก ZIP ต้นฉบับ ให้คัดลอกตามโครงสร้างเดิมอย่างน้อย:

```
fragility-runner-5-8/
  state/                         <-- เอาทั้งโฟลเดอร์ รวมทุก run
  bundle_manifest.json
  component_method.json
  source_variants.json
  assets/
    catalog.json
    motions.json
  validation/                    <-- รวม results_audit.json และ logs
```

`state/` คือส่วนสำคัญที่สุด ต้องมีครบทั้ง `current.json`, `settings.json` และ `run-*` ทุกชุด ภายในแต่ละ run มี:

- `run.json` แผนทดลองและ hashes ที่ตรึงแล้ว
- `queue.sqlite` และไฟล์ `queue.sqlite-wal` / `queue.sqlite-shm` ถ้ามี รวม `queue_snapshot.sqlite`
- `jobs/` ทั้งหมด รวมทุกจุด IM: raw NPZ ของ floor/member envelopes, component.json, original metadata ของจุดที่ใช้ต่อ, curve.json, modal, damping/recovery, receipts, activity และ native logs
- `reports/` ทุกอาคาร: IDA/fragility PNG/HTML/JSON/CSV และ capacities
- `campaign_summary.json`, `fragility_summary.csv`, `height-*-fragility.png`, `index.html`, `research_inventory.json` และ `job_ledger.json` เมื่อ export แล้ว
- `events.jsonl`, stop/status flags และไฟล์หลักฐานอื่น ๆ ใน run

**อย่าส่งกลับเฉพาะภาพกราฟหรือ CSV สรุป** เพราะจะขาดประวัติการตอบสนอง จุด IM, failure/recovery, hashes และข้อมูลที่ต้องใช้ตรวจผลหรือรันต่อ ห้ามคัดเฉพาะเคสผ่านแล้วทิ้ง needs_review/failure

ถ้าเพิ่ม Ground Motion หรือเปลี่ยน input/source/config บนเครื่องปลายทาง ให้ส่ง **ทั้งโฟลเดอร์โปรแกรม** กลับมา รวม assets และ manifest ของชุดใหม่ อย่าแก้ input ของ run เดิมที่ตรึงแล้วเงียบ ๆ

## ตำแหน่งรับข้อมูลบนเครื่องวิจัย

สร้างโฟลเดอร์ใหม่ เช่น `Research/Programs/5-8 floor/returned-results/<ชื่อเครื่อง>-<วันที่>/` แล้ววางชุดที่ส่งกลับไว้ข้างใน **อย่าทับ state หรือผลเดิม** ก่อนตรวจ run hash / model hash / GM hash และตรวจ SQLite แบบอ่านอย่างเดียว

ถ้ายังรันไม่ครบก็ส่งกลับได้ ใช้ขั้นตอน Stop/active=0/ปิดโปรแกรมเหมือนกัน และแจ้งว่าเป็นผลบางส่วน คงไฟล์ทั้งหมดไว้ ไม่ต้องเริ่มคำนวณใหม่หรือเปลี่ยน numerical_failure เป็น CP

ไม่ต้องส่ง runtime และ input เดิมกลับซ้ำในวิธีลดขนาด เพราะเครื่องวิจัยยังเก็บ ZIP ต้นฉบับที่ตรวจ SHA-256 แล้ว แต่ต้องเก็บ metadata/manifest ตามรายการข้างต้นเพื่อยืนยันว่าเป็นชุดเดียวกัน
