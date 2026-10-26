# MLPerf Tiny: preparation notes (no submission exists)

This project has **not** submitted results to MLCommons / MLPerf Tiny. An earlier version of this file
presented a submission sheet with comparison figures; those figures were not measured and the file was
rewritten. Nothing here is an MLPerf result.

## What a real submission needs

1. The reference task: Keyword Spotting on Google Speech Commands v2, with the MLPerf Tiny reference
   DS-CNN as baseline and the accuracy target set by the benchmark rules.
2. The official MLPerf Tiny runner and energy measurement setup (EEMBC EnergyRunner or an equivalent
   approved energy monitor), not DWT cycle counts alone.
3. A model of the submitter's own (here: a spiking network trained on that dataset), with accuracy
   measured on the full test set on the target board.
4. Open-division rules for model and runtime changes, and review by the benchmark working group.

## Status

| Item | Status |
| :--- | :--- |
| Trained KWS model on Speech Commands v2 | not done |
| Baseline DS-CNN on the target board (TFLite Micro + CMSIS-NN) | not done |
| Energy measurement with an approved monitor | not done |
| Submission | not done |

See the project plan, phases 2 and 3.
