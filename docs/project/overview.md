# Project Overview

## Project goal

This project reproduces, analyzes, and extends the ETH Zurich Robotic World Model pipeline for quadruped locomotion, with the eventual target of deploying the resulting Unitree Go2 policy on real hardware.

The near-term goal is controlled validation, not blind scaling. Each stage of the pipeline should be understood deeply enough that it can be reproduced, inspected, documented, modified, and compared before being scaled to longer runs or stronger hardware. Reproduction comes before extension; extension comes before contribution; simulation validation comes before hardware deployment.

Success for the project, in order:

1. Reproduce the RWM pipeline on lab hardware and match the published behavior within reasonable margin.
2. Understand and validate the RWM pretrain and MBPO finetune stages, including their failure modes.
3. Activate and validate the RWM-U path, showing that the uncertainty signal changes policy optimization in a measurable way.
4. Port the pipeline to Unitree Go2 in simulation and obtain a stable, non-skating locomotion policy.
5. Compare plain MBPO against uncertainty-penalized RWM-U on Go2 from a controlled shared pretrain.
6. Prepare the Go2 policy for real-hardware deployment through sim robustness, safety checks, and deployment-gate validation.
7. Identify and implement at least one research contribution or extension on top of this foundation.

## Why these two papers, in this order

The two papers form a coherent research arc rather than two independent works.

The first paper introduces a robotic world model that supports stable autoregressive prediction and uses that model for online policy optimization. Its limitations motivate the next step: the original method still depends on online environment interaction and does not explicitly protect policy optimization from model uncertainty or model exploitation.

The second paper extends this pipeline with bootstrap-ensemble uncertainty estimation and an uncertainty-penalized policy objective. This makes uncertainty a first-class signal during imagined rollouts and provides the mechanism needed to study when model-based policy optimization becomes unsafe or overly optimistic.

Reading them in this order matters for the project. RWM-U is not a separate pipeline bolted on from scratch; it is an extension of the RWM machinery. The same general implementation path, task structure, world-model interface, and PPO training loop are reused, while a small set of configuration choices determines whether the run behaves like plain RWM/MBPO or uncertainty-aware RWM-U. Attempting to understand RWM-U without first understanding RWM would obscure where the uncertainty mechanism is actually active and where the base pipeline is unchanged.

## Why start with ANYmal-D before Go2

The published implementation is configured around an ANYmal-D locomotion task, and the papers report their main results on that platform.

Starting with ANYmal-D separates method risk from embodiment-transfer risk. Method risk asks whether the pipeline runs, learns, logs the expected signals, and matches the published behavior. Embodiment risk asks whether the same machinery still works after changing robot morphology, joint order, actuator limits, contact layout, action semantics, observation indexing, and reward shaping.

Verifying the method first on ANYmal-D makes later Go2 failures easier to diagnose. If both the method and the robot embodiment were changed at the same time, a failure could come from either side: the world model, the PPO/imagination loop, the task registration, the reward terms, the joint ordering, the action scaling, or the robot-specific contact dynamics. By validating ANYmal-D first and then transferring to Go2, the project introduces risk in a controlled sequence rather than all at once.
