# Paper Reading Notes

## Purpose
This file is the working notebook for the paper-to-code mapping phase.

It is intentionally structured but currently only initialized.

---

## Paper A
### Identifier
Robotic World Model / base paper

### Core question
To be filled during P1.

### Claimed contribution
To be filled during P1.

### Key equations
To be filled during P1.

### World-model components
To be filled during P1.

### Uncertainty mechanism
To be filled during P1.

### Training logic
To be filled during P1.

### Code mapping hypotheses
To be filled during P1.

---

## Paper B
### Identifier
Related lightweight / uncertainty-aware stack paper

### Core question
To be filled during P1.

### Claimed contribution
To be filled during P1.

### Key equations
To be filled during P1.

### World-model components
To be filled during P1.

### Uncertainty mechanism
To be filled during P1.

### Training logic
To be filled during P1.

### Code mapping hypotheses
To be filled during P1.

---

## Cross-paper mapping questions
- Which ideas are actually present in the released code?
- Which ideas are simplified or omitted?
- Where do the losses appear concretely?
- Which components are upstream Isaac/RSL abstractions rather than paper-specific contributions?
- Which embodiment assumptions are tied to ANYmal-D and likely to break for Go2?

---

## Implementation mapping checklist
- runner entrypoint
- environment config
- policy config
- world-model module
- uncertainty module
- replay/dataset logic
- training loss definitions
- rollout/imagination logic
- evaluation path\n