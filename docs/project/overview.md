# Project Overview

## Project goal

The project reproduces, analyzes, and extends the ETH Zurich Robotic World Model pipeline for quadruped locomotion, with the eventual target of deploying it on the Unitree Go2 platform on real hardware.

The near-term goal is not to run training at scale. The near-term goal is to understand the published implementation deeply enough that each stage can be validated, documented, modified, and later scaled to stronger hardware. Reproduction comes before extension; extension comes before contribution.

Success for the project, in order:

1. The RWM pipeline is reproduced on lab hardware and matches the published results within reasonable margin.
2. The RWM-U pipeline is activated, validated, and produces a policy that meaningfully exploits the uncertainty signal.
3. The pipeline is ported to the Unitree Go2 platform in simulation.
4. The Go2 pipeline is deployed on real hardware.
5. At least one extension or contribution is identified and implemented on top of this foundation.

## Why these two papers, in this order

The two papers form a coherent research arc rather than two independent works.

The first paper introduces a world model that supports stable autoregressive prediction and uses it for online policy optimization. The paper's own stated limitations identify two gaps: training requires online environment interaction, and the method has no explicit uncertainty handling. The second paper addresses both. It augments the world model with bootstrap-ensemble uncertainty estimation that propagates through long rollouts, then combines it with an uncertainty-penalized PPO objective to make the pipeline work in the fully offline setting.

Reading them in this order matters for the project. The RWM-U codebase, machinery, and configuration are extensions of RWM rather than a parallel approach. The same upstream codebase implements both, with a small set of configuration parameters switching between the two paths. Attempting to understand RWM-U without first understanding RWM produces analysis that misses where the extension is and is not active.

## Why start with ANYmal-D before Go2

The published implementation is configured around an ANYmal-D locomotion task. Both papers report their main results on this platform.

Starting with ANYmal-D separates two sources of risk. Method risk (does the pipeline run, does it learn, does it match the paper) is verified on the platform the paper targets. Embodiment risk (joint ordering, action ranges, sensor differences, reward shaping) is then introduced separately when the project transitions to Go2. If both were tackled at once, debugging a failure would require disentangling whether the issue is in the method or in the embodiment transfer, which is more costly than addressing them sequentially.

## What this project is not

Stating the scope boundaries explicitly avoids confusion about what the work commits to.

- The project does not contribute new methods to RWM or RWM-U as published. The contribution phase comes after reproduction, extension, and Go2 deployment; the form of the contribution is intentionally not fixed in advance.
- The project does not currently train from scratch on a Go2 simulation. ANYmal-D is the validation platform until the embodiment transition is planned.
- The project does not modify Isaac Lab. The submodule strategy treats it as upstream infrastructure, with project-specific changes confined to the two forked codebases that do warrant modification.