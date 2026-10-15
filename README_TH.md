# โปรแกรม Component IDA / Fragility ชั้น 5–8

พร้อมคัดลอกทั้งโฟลเดอร์ไป Windows x64 อีกเครื่อง มี Python, OpenSees, โมเดล 2,000 อาคาร, source, CMS และข้อมูลที่นำมาใช้ต่ออยู่ภายใน ไม่ต้องติดตั้ง Python เพิ่ม การเปิดโปรแกรมไม่เริ่ม solver จนกด Start และแพ็กเกจนี้ไม่ได้รัน population บนเครื่องที่เตรียมงาน

## เริ่มใช้งานที่เครื่องปลายทาง

1. แตก ZIP ให้ครบ หรือคัดลอกโฟลเดอร์ `fragility-runner-5-8` ทั้งโฟลเดอร์ไปตำแหน่งที่มีพื้นที่พอ
2. เปิด `VERIFY_PACKAGE.bat` เพื่อตรวจไฟล์, hash, catalog order และโมเดล X/Y แบบไม่วิเคราะห์
3. เปิด `START_FRAGILITY.bat` แล้วเข้า dashboard ที่ `http://127.0.0.1:8788`
4. เลือกจำนวน workers แล้วกด **Start / Resume** ค่าเริ่มต้น 7 ปรับได้ตามเครื่อง คิวเติมทันทีเมื่อ worker ว่าง ไม่มีการรอจบ batch
5. **Stop safely** หยุดส่งงานใหม่ แล้วให้แต่ละ worker จบจุด IM ปัจจุบันก่อนพัก หากจะย้ายเครื่อง ให้รอ active = 0 ปิดโปรแกรม แล้วคัดลอกทั้งโฟลเดอร์รวม `state`

Dashboard แสดงรายชั้นและรวม 2,000 อาคาร, จำนวน processed / certified / needs review, งานแต่ละ worker, axis / IM / record %, modal / component curves / reports และพื้นที่ว่าง Processed นับงานที่ต้องตรวจด้วย จึงมี certified % แยกต่างหาก

## หนึ่ง component = หนึ่งเส้น IDA

ชั้นละ 500 model definitions เดิมทุกไบต์ ชั้น 5 ใช้ **mesh16 ล่าสุด** ตามที่เลือก ส่วนชั้น 6–8 ใช้ชุดส่งมอบเดิม รวม 2,000 อาคาร ไม่เปลี่ยนวัสดุ หน้าตัด มวล โหลด mesh หรือจูนกลศาสตร์ใหม่ Source ชั้น 5 ที่แตกต่างจาก 6–8 จำนวน 5 ไฟล์ถูกเก็บคนละ snapshot และเลือกตามชั้นใน worker แต่ละ process

มี adapter เฉพาะ I/O เพื่อให้ helper ชั้น 5 เดิมใช้ runtime/cache ภายในแพ็กเกจปัจจุบันและย้ายไปไดรฟ์ C: ได้ แทนข้อจำกัด D: ของเครื่องวิจัยเดิม ไฟล์ต้นฉบับและ numerical physics ไม่ถูกแก้

เลือก CMS family เดียวที่ใกล้ `sqrt(T1X*T1Y)` ที่สุด ไม่มี midpoint blend ใช้ชุด CMS เดิม 23 คู่ รวม family T*=3.0 s ไม่แทนด้วยชุด Fig.5 ที่ยังทำไม่เสร็จ Family 3 คู่ได้ 6 components/IDA และ family 4 คู่ได้ 8 components/IDA

- X waveform ใช้โมเดลทิศ X และ Y waveform ใช้โมเดลทิศ Y เป็นคนละงาน
- แต่ละเส้นมี `IM = Sa_component(T1_direction, 5%)` และ `SF = IM / Sa_component_unscaled` ของตัวเอง
- ไม่รวม max drift สองทิศ และไม่ใช้ geometric-mean Sa ของคู่เป็น IM
- IO/LS/CP คือ MIDR 0.01/0.02/0.04 ตามเดิม แต่ละ component หา capacity และ lower/upper bracket ของตนเองให้กว้างไม่เกิน 5%
- Initial IM .05 g, hunt ×1.5, log bisection; guard .001–20 g / 100 จุด / SF 10,000 / 7,200 s ต่อ direction ไม่ใช่ capacity เมื่อถึง guard หรือ numerical failure จะเก็บ needs_review และไม่ retry แบบเดิมอัตโนมัติ

## ข้อมูลชั้น 5 ที่นำมาด้วย

มี latest mesh16 directional histories เดิม 192 จุดพร้อม original metadata/flags/hash จาก Development 8 อาคาร ในจำนวนนี้ **72 จุดของ 3 อาคารมี ID ตรงกับ catalog 500 อาคาร** ก่อนเลือก family จึงไม่อ้างว่ามีผล IDA mesh16 ครบทั้ง 500 อาคารแล้ว

เมื่อตรงทั้ง building ID, candidate, component, waveform hash และ T1 โปรแกรมจะนำผลที่สมบูรณ์หรือมี raw CP endpoint ของทิศนั้นมาใช้ต่อ และคำนวณ component IM ใหม่จาก scale factor ที่ใช้จริง ตรวจ paired Sa เดิมเทียบกับ waveform snapshot ด้วย ไม่ใช้ partial non-CP prefix เป็นจุดสำเร็จ ไม่เปลี่ยน numerical failure เป็น CP และไม่รัน signed failure ซ้ำอัตโนมัติ

ไฟล์ `legacy_import.json` ของแต่ละ component บันทึกจำนวนที่ใช้/ไม่ใช้ จุดที่ใช้ต่อมี original metadata และ raw NPZ hash ใน `component.json` / CSV ผลเดิมที่ไม่ได้บันทึก member envelopes จะระบุว่าขาด ไม่สร้างข้อมูลย้อนหลังขึ้นมา ผล mesh8 อยู่เป็นประวัติเดิมและไม่ปนกับ primary mesh16

รวมชุดรายงานและ comparison evidence ล่าสุดของชั้น 5–8 ไว้ใน `assets/models/<ชั้น>storey` รวมทั้ง static/cyclic attempts และ failures เดิม รายงาน Development เดิมเป็น paired reference จึงไม่อ้างว่าเป็นการตรวจ Full3D แบบเขย่าทิศเดียวใหม่ หรือเป็น general GM accuracy / physical collapse

## ผลวิจัยที่จะบันทึก

แต่ละ IM เก็บ raw floor histories ทุก committed state: displacement, story deformation/drift, absolute acceleration และ inertia-based base shear รวมถึง member localForce/basicDeformation min/max/absmax envelopes, modal/gravity, damping, scale factor, recovery, runtime, logs และ hashes ไม่ใช่ full member/fiber time histories

มี component IDA/capacities และ fragility รวม components/แยก X/Y ต่ออาคาร เป็น HTML/PNG/JSON/CSV/NPZ พร้อม inventory ความไม่แน่นอน bootstrap สุ่มเป็นกลุ่ม parent pair เพราะ 6–8 components ไม่ใช่ 6–8 เหตุการณ์อิสระ จำนวน parent pairs เพียง 3–4 เป็นข้อจำกัดของช่วงความเชื่อมั่น

Final fit ของอาคารสร้างเมื่อทุก component ที่เลือกครบเท่านั้น ส่วนภาพรวมรายชั้นรอครบทั้ง 500 อาคารและให้น้ำหนักแต่ละอาคารเท่ากัน ข้อมูลที่ติด guard/failure ยังคงอยู่แต่ไม่ถูกนับเป็น certified fragility เปิด `VERIFY_RESULTS.bat` เพื่อตรวจหลักฐานหลังรัน

## ความต่อเนื่องและการย้ายเครื่อง

แผนทดลอง, model/recipe/GM/source hashes และ case signatures ถูกตรึงใน run แต่ละ point ใช้ต่อได้เมื่อ commit/hash/ตัวตนตรงเท่านั้น ไม่ใช้ partial prefix ถ้า worker เดิมยังมีชีวิต การเปิด dashboard ใหม่จะตรวจ executable + PID + creation time เพื่อไม่เปิดงานซ้ำ เปลี่ยน worker count แล้วไม่ฆ่างานที่กำลังคำนวณ

ไม่เริ่ม calibration, Full3D, ชั้น 9–12 หรือชั้น 20 และไม่มี worker แอบทำงานเมื่อเปิดหน้าเริ่มต้น หากเพิ่ม GM ในอนาคตต้องประกาศ extension แยกและตรวจความเข้ากันของ model/IM/physics/hash ก่อนนำจุดเดิมมาใช้; ห้ามเปลี่ยน input ของ run ที่ตรึงแล้วเงียบ ๆ
