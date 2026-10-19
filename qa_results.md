## QA run 2026-10-13 23:24

| Scenario | Expect | Result | Detail | Face found | Trusted | EAR med/min | Bright | Sharp | Max level | Issues |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | quiet | PASS | no alarm | 100% | 100% | 0.27/0.06 | 100 | 519 | OK | - |
| Talking | quiet | PASS | no alarm | 100% | 100% | 0.23/0.09 | 100 | 496 | OK | - |
| Mirror check | quiet | PASS | no alarm | 100% | 81% | 0.28/0.06 | 104 | 483 | OK | - |
| Yawn | yawn | PASS | 1 yawn(s) counted | 100% | 95% | 0.16/0.11 | 103 | 716 | CRITICAL | - |
| Eyes closed | alarm | PASS | alarm after 1.2s (includes your reaction time) | 100% | 100% | 0.07/0.04 | 103 | 586 | CRITICAL | - |
| Head drop | alarm | FAIL | NO ALARM | 50% | 100% | 0.27/0.16 | 102 | 369 | OK | no_face |
| Touch your face | no_phone | FAIL | FALSE PHONE CALL | 100% | 100% | 0.27/0.13 | 102 | 400 | WARNING | - |
| Phone call | phone | PASS | phone pose recognised after 2.2s | 100% | 100% | 0.28/0.26 | 102 | 444 | WARNING | - |

**6/8 scenarios passed.**

