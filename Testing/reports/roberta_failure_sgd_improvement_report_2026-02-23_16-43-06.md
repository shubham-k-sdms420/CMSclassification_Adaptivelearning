# RoBERTa Failure vs SGD Improvement — Manager Report

**Generated:** 2026-02-23 16:47:00

**Scope:** Weak categories only (see `Testing/weak_categories.txt` and `category_mapping.csv`).

---

## Purpose

This report lists **only** complaints where **RoBERTa's prediction was wrong** (different from the human-corrected category) and **SGD corrected it** after learning from feedback. For each such complaint we show: **original complaint**, **RoBERTa model prediction** (wrong), **SGD corrected category** (human feedback that SGD learnt), and **learnt status**. This proves where RoBERTa failed and how SGD improved using feedback.

---

## Summary

- **Complaints shown (RoBERTa wrong + SGD corrected):** 24
- These are weak-category complaints where RoBERTa's prediction differed from the corrected category and SGD learnt the correct category from feedback.
- (From 47 weak-category feedback records, 24 met the "RoBERTa wrong, SGD corrected" criteria.)

---

## Detailed Table

| Original complaint | RoBERTa model prediction | SGD corrected category | Learnt status |
|:-------------------|:-------------------------|:----------------------|:-------------:|
| Issue with garbage depot complaint at Camp area. staff not cooperative. This is ... | City Development Plan | Garbage Depot Complaint | Yes |
| Problem in unauthorized banners / advertisements / permit / license in pune city... | Unauthorized hoardigs banners / advertisements on roads / footpath / buildings / directions panels | Unauthorized banners / advertisements / Permit / license in Pune City | Yes |
| Sir/Madam, unauthorized banners / advertisements / permit / license in pune city... | Unauthorized hoardigs banners / advertisements on roads / footpath / buildings / directions panels | Unauthorized banners / advertisements / Permit / license in Pune City | Yes |
| License (Parwana) application pending at Mundhwa. Take immediate action. Referen... | Bogus doctor / pregnancy diagnosis / unauthorized sonography center | License (Parwana) | Yes |
| Complaint regarding license (parwana) at Sinhagad Road. online system not workin... | Encroachments on public premises / roads | License (Parwana) | Yes |
| Residents of Camp facing issue with unauthorized banners / advertisements / perm... | Unauthorized hoardigs banners / advertisements on roads / footpath / buildings / directions panels | Unauthorized banners / advertisements / Permit / license in Pune City | Yes |
| Unauthorized banners / advertisements / Permit / license in Pune City at Akurdi ... | Unauthorized hoardigs banners / advertisements on roads / footpath / buildings / directions panels | Unauthorized banners / advertisements / Permit / license in Pune City | Yes |
| At Deccan, garbage depot complaint no proper facility. Please resolve urgently. | City Development Plan | Garbage Depot Complaint | Yes |
| Urgent: license (parwana) not maintained at Viman Nagar. Residents are suffering... | Encroachments on public premises / roads | License (Parwana) | Yes |
| At Khadki, license (parwana) poor quality. Take immediate action. Reference: PMC... | RTI Related Complaint | License (Parwana) | Yes |
| Garbage Depot Complaint at Bavdhan needs attention. online system not working. N... | Information Technology | Garbage Depot Complaint | Yes |
| Issue with garbage depot complaint at Balewadi area. no proper facility. Need qu... | City Development Plan | Garbage Depot Complaint | Yes |
| Sir/Madam, information technology at Kondhwa poor quality. Need PMC intervention... | Information Technology-Employee | Information Technology | Yes |
| Problem in birth and death department at Pimple Gurav. online system not working... | Public Health Related | Birth And Death | Yes |
| Request for birth and death at Sinhagad Road. Currently poor quality. This is af... | Public Health Related | Birth And Death | Yes |
| Complaint regarding birth and death at Kalyani Nagar. no response. Business affe... | Public Health Related | Birth And Death | Yes |
| Information Technology health hazard at Chakan. Please resolve urgently. Previou... | Information Technology-Employee | Information Technology | Yes |
| Garbage Depot Complaint urgent attention needed at Kondhwa. Request prompt atten... | Road Sweeping / Toilet Cleaning / Garbage disposal | Garbage Depot Complaint | Yes |
| City Development Plan at FC Road needs attention. irregular service. Request pro... | Building Permission | City Development Plan | Yes |
| At Pimpri, public health related application pending. Residents are suffering. | Aids Control | Public Health Related | Yes |
| Urgent: information technology no response at Hadapsar. Need quick solution. Ref... | RTI Related Complaint | Information Technology | Yes |
| Urgent: public health related facing difficulty at Dapodi. Request prompt attent... | Aids Control | Public Health Related | Yes |
| Issue with license (parwana) at Magarpatta area. delayed service. Please don't d... | Employee Transfer / Promotion / Service Record/Others (GAD/Est.)/Regarding Officer/Women Harassment | License (Parwana) | Yes |
| At Camp, garbage depot complaint issue since 45 days. Take immediate action. Ref... | City Development Plan | Garbage Depot Complaint | Yes |

---

## Glossary

- **Original complaint:** Excerpt of the complaint text.
- **RoBERTa model prediction:** The (wrong) category RoBERTa predicted for this complaint; here it always differs from the corrected category.
- **SGD corrected category:** The category provided as human feedback; the target label SGD learns from.
- **Learnt status:** Yes = the system has stored this feedback (same complaint would show corrected category on re-submission).

- **Final prediction (current):** The ensemble’s final label (combines RoBERTa + SGD); when learnt, this is the corrected category.
- **RoBERTa failed?:** Yes = RoBERTa’s current prediction is different from the corrected category.
- **SGD improved?:** Yes = SGD’s current prediction matches the corrected category (SGD learnt from feedback).

---

*Report generated by roberta_failure_sgd_improvement_report.py — 2026-02-23_16-43-06*
