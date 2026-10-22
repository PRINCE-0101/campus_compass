# Dataset card

216 original AI-assisted examples: 36 each for academics, events, transport, clubs, facilities and other. These are authored synthetic queries, not collected from students and not copied from CLINC150. No private student data is included. English only.

## Why this route
The supplied screening task explicitly permits writing your own queries (at least 10 per category), mapping CLINC150 intents, or both. This submission takes the original-query route. CLINC150 is a general intent resource, not a campus-labeled corpus; forcing unrelated intents into academics/clubs would introduce label noise.

Reference supplied by the task: https://huggingface.co/datasets/DeepPavlov/clinc150 (reviewed 2026-10-22). No claim is made that the model is trained or evaluated on CLINC150. If the club separately requires actual CLINC150 use, this version needs a documented additional data experiment.

## Annotation policy
| Label | Meaning | Boundary |
|---|---|---|
| academics | Courses, exams, grades, faculty, educational records, tuition | A technical seminar is events, not academics |
| events | Time-bound competitions, talks, fests, attendance/registration | Joining the organising club is clubs |
| transport | Bus, shuttle, routes, parking, campus mobility | Vehicle issue is transport; hostel repair is facilities |
| clubs | Membership, recruitment, leadership, society projects | Attending a society's public workshop is events |
| facilities | Infrastructure, rooms, library access, WiFi, hostel, health centre | Library bus route is transport |
| other | General questions outside campus routing scope | Also the inference fallback, kept separate via a reason field |

Mixed-intent questions have no supported multi-label policy; ask users to split them. The model does not answer factual campus queries.

## Reproducible construction
`queries.json` is the editable source. `python train.py` normalizes text, rejects conflicting labels/empty cleaned rows, removes exact normalized duplicates and creates `campus_queries.csv`. A stratified split with seed 42 reserves approximately 60% training, 20% validation, 20% test. Integer rounding gives 129/43/44 rows. Source provenance and split are retained in CSV. The audit records counts and a SHA-256 source fingerprint.

No template expansion is used. Similar topics may exist across partitions; exact deduplication does not establish independence between paraphrases. Do not present these scores as a production benchmark. Collect an independently authored, consented evaluation set before deployment. Preserve a new untouched test set when iterating on the present errors.
