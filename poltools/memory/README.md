# poltools analyst — usage

## Workflow A (preferred): fresh chat per question

```
@analyst-poltools @poltools/_types.py

What is plate_scale_arcsec in PolConfig, and why is the default 0.224?
```

```
@analyst-poltools @poltools/mueller.py

What is the reference for the matrix methods used?
```

Attach **only** the skill + the file you're asking about. The router lives inside the skill. LEDGER is grepped automatically. Leaf files (SCIENCE, INSTRUMENT, QUIRKS) are read on demand.

## Workflow B: batch 5–8 questions

```
Q1: @analyst-poltools @poltools/_types.py
    <question>

Q2: @poltools/mueller.py
    <question>
```

New chat after Q8. Re-invoke `@analyst-poltools` on the first question of the new batch.

## Do not attach

| File | Why |
|------|-----|
| `INDEX.md` | Embedded in skill |
| `LEDGER.md` | Skill greps it every turn |
| `SCIENCE.md` etc. | Skill reads on demand |

## Skip `prepare`

Goes straight to the question. No separate bootstrap step.
