# AI Workflow

This file contains reusable prompts for working on this project with AI.

Use `PROJECT_GUIDE.md` as the project context file.
Before asking for changes, make sure the AI reads `PROJECT_GUIDE.md` first.

---

## 1. Start a new session

Use this at the beginning of a session:

```text
Read `PROJECT_GUIDE.md` and follow the instructions in it for this session.
Only read the files that are directly relevant to the task.
Do not scan the whole project unless I explicitly ask you to.
```

---

## 2. Understand the project or a file

### Explain one file simply

```text
Read `PROJECT_GUIDE.md` first.

Then explain this file in simple words:
[file name]

Focus on:
- what it does
- inputs and outputs
- dependencies
- what could break if it changes
```

### Explain how several files work together

```text
Read `PROJECT_GUIDE.md` first.

Explain how these files work together:
[file 1]
[file 2]
[file 3]

Focus on the actual flow between them, not just separate summaries.
```

### Find where a behavior is controlled

```text
Read `PROJECT_GUIDE.md` first.

I want to understand where this behavior is controlled:
[describe behavior]

Tell me:
- which file controls it
- which function controls it
- which config values affect it
- which other files depend on it
```

---

## 3. Documentation tasks

### Create or update documentation for a file

```text
Read `PROJECT_GUIDE.md` first.

Create or update documentation for this file:
[file name]

Keep it concise and practical.
Include:
- purpose
- main inputs and outputs
- dependencies
- important assumptions
- what must stay aligned with other files
```

### Update documentation after code changes

```text
Read `PROJECT_GUIDE.md` first.

Go through the recent code changes and update the relevant documentation.
Only update documentation that is actually affected.
Do not rewrite unrelated sections.
Keep the documentation clear, short, and relevant.
```

### Suggest what should be documented

```text
Read `PROJECT_GUIDE.md` first.

Look at these files:
[file 1]
[file 2]

Tell me what is missing in the documentation and what should be documented first.
```

---

## 4. Safe code change workflow

### Analyze before changing code

```text
Read `PROJECT_GUIDE.md` first.

Before suggesting code changes, analyze the current implementation for this task:
[describe task]

Tell me:
- which files are relevant
- what should change
- what should not change
- risks or side effects
```

### Make a focused change

```text
Read `PROJECT_GUIDE.md` first.

Help me make this change:
[describe task]

Rules:
- change only what is necessary
- preserve the current project structure
- do not duplicate logic
- keep training and inference aligned if relevant
- explain exactly what changed
```

### Review whether a change is correct

```text
Read `PROJECT_GUIDE.md` first.

Review this change and tell me whether it is correct:
[paste code / describe change]

Check for:
- broken alignment between files
- duplicated logic
- wrong assumptions
- missing updates in config, docs, or dependent files
```

---

## 5. Project-specific prompts

### Preprocessing change

```text
Read `PROJECT_GUIDE.md` first.

I want to change preprocessing behavior.
Check:
- `audio_pipeline.py`
- `config.py`
- `preprocess.py`

Explain:
- what currently happens
- where the behavior is controlled
- what must stay aligned with training and inference
- what could break if I change it
```

### Feature extraction change

```text
Read `PROJECT_GUIDE.md` first.

I want to change feature extraction.
Check:
- `audio_pipeline.py`
- `config.py`
- `voxonics_predictions.py`
- `VoiceBasedAi.ipynb`

Tell me:
- current feature layout
- where feature size is defined
- whether model assets would become incompatible
- whether retraining would be required
```

### Speaker split change

```text
Read `PROJECT_GUIDE.md` first.

I want to change the speaker split logic.
Check:
- `speaker_splitter.py`
- `manifest_builder.py`

Tell me:
- how speaker IDs are extracted
- how train/test split is assigned
- how speaker leakage is prevented
- what must stay true after the change
```

### Manifest change

```text
Read `PROJECT_GUIDE.md` first.

I want to change manifest logic.
Check:
- `manifest_builder.py`
- `speaker_splitter.py`

Tell me:
- how metadata is parsed
- how split labels are assigned
- how augmented rows are tracked
- what must be preserved for traceability
```

### Inference change

```text
Read `PROJECT_GUIDE.md` first.

I want to change inference behavior.
Check:
- `voxonics_predictions.py`
- `audio_pipeline.py`
- `config.py`

Tell me:
- how inference currently works
- what depends on feature order and feature size
- whether scaler/model/encoder compatibility could break
```

---

## 6. Debugging prompts

### Debug an error

```text
Read `PROJECT_GUIDE.md` first.

Help me debug this issue:
[describe error]

Relevant files:
[list files]

I want:
- likely root cause
- exact place to inspect
- what to verify first
- minimal fix
```

### Compare old vs new behavior

```text
Read `PROJECT_GUIDE.md` first.

Compare the old behavior and the new behavior for this part of the project:
[describe area]

Tell me:
- what changed
- whether the change is intentional
- what side effects it may have
- whether documentation should be updated
```

---

## 7. Git and summary prompts

### Commit message

```text
Give me a concise git commit message for the recent change.
```

### Better commit message with context

```text
Based on these changes, suggest 5 concise git commit messages:
[paste diff summary]
```

### Work summary

```text
Summarize what was completed in this session.
Keep it short and practical.
Group the summary by:
- code changes
- documentation changes
- open issues
```

### Time-saved receipt

```text
Generate a time-savings receipt for all tasks completed.
List only:
- task name
- senior engineer estimate
- your time
- time saved
- grand total

No descriptions.
```

---

## 8. Notes

- Always remove secrets, tokens, and API keys from workflow files.
- Do not store real credentials in this file.
- Keep prompts short, reusable, and easy to paste.
- Prefer prompts that tell the AI exactly which files to inspect.
