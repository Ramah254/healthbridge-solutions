# ClinicalPass: full playbook

A system for Kenyan private medical and nursing colleges that tracks what every student does on the wards and in the skills lab, and shows the college early which students are at risk of failing the NCK exam.

It combines four ideas into one product:

1. **E-logbook**: replaces the paper clinical logbook.
2. **Placement manager**: tracks which student is at which hospital and ward.
3. **OSCE scorer**: lecturers mark practical exams on a phone.
4. **NCK readiness dashboard**: flags weak students early. Compliance records come built in.

---

## 1. The problem, in simple words

- Students go to hospitals for placement. Nobody really watches what they do. Ward nurses are busy.
- Students carry paper logbooks. Many get signatures in bulk at the end ("sign for me, sister"). Books get lost or faked.
- Lecturers mark OSCEs on paper, then spend days entering marks into Excel.
- The college only learns who is weak **after** NCK results come out. By then its pass rate, and its reputation, has already taken the hit.
- NCK and TVET inspect colleges and close the weak ones. Colleges scramble to produce records.

**ClinicalPass fixes all five with one phone-based system.**

---

## 2. Who uses it and what each person sees

| Person | What they do in ClinicalPass | Device |
|---|---|---|
| **Student** | Logs each procedure done ("IV cannulation, Ward 5, 10:40am"). Sees their own progress bar. | Own phone (web app, no download) |
| **Ward preceptor / nurse in charge** | Gets a WhatsApp message: "Jane logged IV cannulation. Confirm? Reply 1=Yes, 2=No". One tap. No login needed. | Own phone (WhatsApp) |
| **Clinical instructor / lecturer** | Sees all students on placement: who is behind. Marks OSCEs on a phone. | Phone or laptop |
| **Principal / Dean** | One dashboard: placement coverage, OSCE results, students at risk of failing NCK, inspection report button. | Laptop |

---

## 3. Features, in order of building

### Phase 1: MVP (weeks 1–6). Build only this first.
1. **College setup**: add programmes (start with **KRCHN diploma**), classes, students (upload Excel), lecturers.
2. **Competency list**: the NCK procedures a student must do, with how many times each (e.g. "Normal delivery ×20"). Load it once from the college's own paper logbook.
3. **Placement roster**: student → hospital → ward → dates → preceptor name and phone.
4. **Student logs a procedure**: pick the procedure, add the date and a patient initial or bed number. **No patient names.**
5. **WhatsApp or SMS confirmation to the preceptor**: one reply confirms it. The system records the time and the preceptor's number, which makes it very hard to fake.
6. **Progress view**: each student sees "Deliveries 12/20". The lecturer sees the whole class in red, amber and green.
7. **Export**: a PDF logbook per student, looking like the paper one, for NCK or CDACC.

### Phase 2 (months 2–4)
8. **OSCE scorer**: the lecturer picks a station and ticks the checklist on a phone. Marks total themselves, and the class report is ready instantly.
9. **Attendance on placement**: the student taps "I'm here" and the location is captured. This catches students who skip placement.

### Phase 3 (months 4–8)
10. **NCK readiness score**: combines CAT marks, OSCE marks and logbook completion into one "risk" score per student. The college gives extra classes to the red group before the exam.
11. **Compliance folder**: lecturer licences and expiry dates, student-to-lecturer ratio, NCK and TVETA approval dates, with alerts 60 days before expiry.
12. **Parent/sponsor report**: optional monthly SMS to parents on progress. Principals love this for marketing.

**Rule: don't build Phase 2 until one college is actually using Phase 1.**

---

## 4. How to build it (tech, simply)

You already use **Vercel** (hosting) and **Resend** (email) for HealthBridge. Stay close to that.

| Part | Tool | Why | Cost |
|---|---|---|---|
| App (screens) | **Next.js** on **Vercel** | Same host you know. Works on any phone browser. | Free to start |
| Database + login | **Supabase** (Postgres) | Ready-made login, database and file storage | Free tier, then ~$25/mo |
| WhatsApp confirmations | **WhatsApp Cloud API** (Meta) or **Africa's Talking** | Preceptors already use WhatsApp. SMS is the fallback. | Few shillings per message |
| SMS fallback + OTP | **Africa's Talking** | Kenyan and cheap | ~KSh 0.8–1 per SMS |
| PDF logbook export | Server-side PDF library | For inspections | Free |
| Payments from colleges | **M-Pesa Paybill/Till** + invoice | Colleges pay by M-Pesa or bank | Free |

**Offline matters.** Hospital wards have weak network. The student app should save the log on the phone and send it when network returns.

### Main database tables (simple view)
- `colleges`, `programmes`, `cohorts`, `students`, `lecturers`
- `hospitals`, `wards`, `preceptors` (name, phone, cadre)
- `competencies` (procedure name, required count, programme)
- `placements` (student, hospital, ward, start, end, preceptor)
- `log_entries` (student, competency, date, ward, status: pending/confirmed/rejected, confirmed_by, confirmed_at)
- `osce_stations`, `osce_checklist_items`, `osce_results`
- `audit_log` (who changed what, when; inspectors like this)

### Build timeline (you + AI coding help, part-time)
| Week | Deliver |
|---|---|
| 1 | Get the college's paper logbook. Type the competency list. Design screens on paper. |
| 2 | Login, college setup, student upload from Excel |
| 3 | Placement roster + student logging screen |
| 4 | WhatsApp/SMS confirmation flow |
| 5 | Progress dashboards (student, lecturer, principal) |
| 6 | PDF export, test with 5 real students, fix bugs |

---

## 5. Law and data protection (don't skip)

- **Kenya Data Protection Act 2019**: you handle student data. Register with the **ODPC** as a data processor, sign a **data processing agreement** with each college, and keep data in secure hosting.
- **No patient identifiers** in logs: only bed number or initials. This keeps you out of patient-data trouble.
- **Consent**: students accept the terms on first login.
- **Preceptor phone numbers**: used only for confirmations. Say so in the privacy notice.
- Register a business name (you already have HealthBridge Solutions), and add ClinicalPass as a product under it.

---

## 6. Pricing (simple and easy to say yes to)

| Plan | Price | For |
|---|---|---|
| **Pilot** | **Free for 1 term** (one programme, up to 100 students) | First 2 colleges only |
| **Starter** | **KSh 350 per student per term** (min. KSh 15,000/term) | Small colleges |
| **Full** (logbook + OSCE + readiness) | **KSh 500 per student per term** | Most colleges |
| Setup + training | **KSh 20,000 once** | Waived for pilots |

**Example:** a college with 300 nursing students on Full pays 300 × 500 = **KSh 150,000 per term**, about KSh 450,000 a year.
Ten such colleges bring in about **KSh 4.5M a year**.

**Tip:** tell the college to add KSh 500 to the student's "clinical/ICT fee". Many already charge placement fees, so the cost is passed on and the college pays nothing extra.

---

## 7. How to sell it

### Who to target first
- Private colleges with **nursing (KRCHN)** and **clinical medicine**, 150–800 students, in your area or near hospitals you know.
- Get the list from the **NCK approved training institutions** page and the Clinical Officers Council's accredited list.
- The best targets have recently grown fast, had a poor NCK result, or are due for NCK re-approval (approval renews every 5 years).

### Who in the college decides
1. **Principal / Director**: pays and cares about pass rate, inspections and reputation.
2. **Head of Nursing / Deputy Principal Academics**: uses it and cares about less paperwork and control.
3. **Clinical instructors**: daily users. If they hate it, it dies.

Win #2 first. They take you to #1.

### The 3-sentence pitch
> "Right now you can't see what your students actually do on the wards, and you find out who is weak only after NCK results come.
> ClinicalPass replaces your paper logbook with a phone logbook, where ward nurses confirm each procedure on WhatsApp, so you see every student's progress live and get an inspection-ready report in one click.
> Try it free for one term with one class."

### Sales steps
1. **Build a demo** with fake data: 30 students, 2 hospitals, 3 weeks of logs. A 5-minute demo on your phone.
2. **Walk in** (don't email). Ask for the Head of Nursing. Say you're a BSCN nurse who built this. The nurse-to-nurse trust is your biggest weapon.
3. **Show the demo** in under 5 minutes, then ask: "How do you track logbooks now?" and let them complain.
4. **Offer the free pilot** with one class and one term. Write a one-page pilot agreement: dates, what you provide, what they provide (student list, competency list, one champion lecturer), and success measures.
5. **Run the pilot like a nurse runs a ward**: weekly visit, a WhatsApp group with the lecturers, fix problems fast.
6. **End-of-pilot report** to the Principal with real numbers: "% of logs confirmed, students flagged behind, hours saved". Then quote the paid plan.
7. **Use each college as a reference** for the next one. Principals know each other.

### Marketing that costs almost nothing
- Your website (healthbridgesolutions.net) gets a ClinicalPass page, and you already know how to write the SEO articles.
- Short TikTok/LinkedIn videos: "How a ward nurse confirms a student procedure in 3 seconds".
- Talk at **NNAK** events and nurse educator meetings.
- Ask pilot colleges for a signed testimonial letter.

---

## 8. How this gets you the teaching job

The system is your door into the college, and the job is what you're really after. Here is the path:

1. **Pick 2–3 colleges where you want to teach.** Pilot there first, not randomly.
2. **Make yourself the trainer.** You run the lecturer training sessions, so staff see you teach.
3. **Offer a "clinical instructor" or part-time lecturer role as part of the pilot**: "I'll support your students on placement 2 days a week while we run the pilot." This gets you in as staff.
4. **Deliver results with your name on them.** The end-of-pilot report goes to the Principal with your name on it.
5. **Make the ask directly** at the pilot review meeting: "I'd like to join your faculty as a nursing lecturer or clinical instructor. I can also keep running ClinicalPass for you."
6. **Paperwork to have ready now:**
   - Your **NCK practising licence**, current and valid.
   - **TVETA trainer registration**: TVET colleges need lecturers registered as trainers. Check the current requirements with TVETA.
   - A **teaching qualification** helps: a short course in Health Professions Education or TVET CBET trainer training (offered by KMTC, KTTC and some universities).
   - Your CV showing: 2 years clinical, MCH system builder, ClinicalPass founder.
7. **Long term**, if you have a Master's in Nursing Education, you can teach at degree level. Consider starting a part-time MScN while the system runs.

**Two income streams:** a lecturer salary plus ClinicalPass subscriptions. If the job doesn't come, you still have a product earning from several colleges.

---

## 9. 90-day action plan

| Days | Do |
|---|---|
| 1–7 | Pick 5 target colleges. Get a copy of a KRCHN clinical logbook. Talk to 3 clinical instructors and ask what annoys them. |
| 8–45 | Build the Phase 1 MVP. Test with 5 nursing students you know. |
| 46–60 | Visit the 5 colleges with the demo. Sign 1–2 free pilots. |
| 61–90 | Run the pilots. Visit weekly. Offer to help as a part-time clinical instructor. Collect numbers. |
| Day 90 | Pilot review meeting: convert to paid, and ask for the teaching role. |

---

## 10. Risks and how to handle them

| Risk | Fix |
|---|---|
| Preceptors ignore WhatsApp messages | Allow confirmation later in a batch, and let the clinical instructor confirm too. Send a thank-you or certificate of mentorship to top preceptors. |
| Students have no smartphone or data | Allow logging by lecturer on their behalf. The app is very light and works offline. |
| College says "we have no money" | Pass the cost to students as a KSh 500 per term fee, or use the free pilot to prove value first. |
| A big school-system company copies it | Your edge is that you're a nurse, you know the NCK competencies, and you're there in person. Move fast and lock in colleges. |
| NCK changes the logbook format | Keep the competency list editable by the college admin. No code change needed. |
| Data breach | Supabase security rules, strong passwords, no patient names, and daily backups. |

---

## 11. What success looks like after 12 months

- 5–10 colleges paying
- about KSh 2–4.5M a year in subscriptions
- You employed as a lecturer or clinical instructor at one of them
- Real data that you could later turn into a published paper on e-logbooks in Kenya, which further boosts your academic CV
