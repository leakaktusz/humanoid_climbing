# Thesis experiment journal — climbing RL


## E1 — 2026-06-30 — climbing env first version (`d8c7089` "climbing first try")
- Change: New `ClimberClimbingMJX` env: 17-DOF humanoid on a meshed wall with two jug holds, plus the first reward (progress, reach, grip, alive, fall).

## E2 — 2026-07-01 — spawn fix (`acfd524` "humanoid not in wall")
- Change: Moved the torso spawn in x so the hands clear the wall but the holds stay in reach.

## E3 — 2026-07-01 — minimal reward (`fd2a865` "just progress, grip and reach")
- Change: Removed the alive bonus, leaving only progress, reach and grip.

## E4 — 2026-07-01 — alive back on (`999be1c` "full reward")
- Change: Re-enabled the alive bonus at +0.1/step.

## E5 — 2026-07-01 — feet grip too (`1df5946` "legs stick to holds as well")
- Change: Extended reach, grip and sticky from the two hands to all four limbs, and added low footholds.

## E6 — 2026-07-01 — payout timeout + surface distance (`46f02dc`)
- Change: Added a 4 s per-hold payout timeout, measured grip surface-to-surface, and softened the sticky constraint.

## E7/1 — 2026-07-02 — no alive bonus (`c9e009b`)
- Change: Turned the alive bonus off again.

## E7/2 — 2026-07-02 — grip observation added (`c9e009b`)
- Change: Made the grip observation report physical grip on any hold, so letting go became learnable.

## E8 — 2026-07-03 — anti-farming redesign (`ff069a7`)
- Change: Replaced the farmable streams with a personal-best height potential and a one-time first-touch grip bonus.

## E9 — 2026-07-03 — scale-up on E8 (`d3e578f` "1.5 scale up grip and reach")
- Change: Scaled all hold weights up by 1.5.

## E10 — 2026-07-03 — revert to streams (`4cebd4a` "revert, and scale grip")
- Change: Reverted to the per-step grip stream and raw height potential, with larger reach and grip weights.

## E11/1 — 2026-07-03 — positive reach + height weights (`e4fb94f` "hold rew positive")
- Change: Flipped reach to a positive proximity bonus on the nearest active hold, with height weights restored.
- Run: https://wandb.ai/lealea/climbing_env_mjx/runs/459lpiu6

## E11/2 — 2026-07-03 — reach to the top of higher holds is higher
- Change: Moved the reach target to the top of the hold and summed it over all holds.

## E12 — 2026-07-06 —  hands  height
- Change: Reset to E7, then added hands and feet height terms, an all-holds reach kernel and a body-relative fall check.
- Run: https://wandb.ai/lealea/climbing_env_mjx/runs/459lpiu6

## E13 — 2026-07-06 — leg term added
- Change: Gave the legs the same height reward as the arms.
- Run: https://wandb.ai/lealea/climbing_env_mjx/runs/v82c538f

## E14 — 2026-07-06 — diff weights (less leg)
- Change: Swept the hands, legs and reach weights across five runs.
1 Hand :x3 Leg :x1    Reach :x1               https://wandb.ai/lealea/climbing_env_mjx/runs/obdhtb6q
2 Hand :x3 Leg :x0.25 Reach :x1.5             https://wandb.ai/lealea/climbing_env_mjx/runs/3l6pn1q3
3 Hand :x4 Leg :x0.05 Reach :x1.5             https://wandb.ai/lealea/climbing_env_mjx/runs/mci1gk4h
4 Same with 60 mil steps:                     https://wandb.ai/lealea/climbing_env_mjx/runs/segmd61j
5 Hand :x5                                    https://wandb.ai/lealea/climbing_env_mjx/runs/8co1d9gk

## E15 — 2026-07-06 — more timeout
- Change: Ended the episode when max height stopped improving for five seconds, with a longer episode cap.

## E16 — 2026-07-08 — revert to E14 / 3
- Change: Reverted to the best E14 variant (E14/3) and trained it longer.
1 https://wandb.ai/lealea/climbing_env_mjx/runs/pw4k5gva
2 https://wandb.ai/lealea/climbing_env_mjx/runs/3eyh4vpd
3 https://wandb.ai/lealea/climbing_env_mjx/runs/jvr76bxz

## E17 — 2026-07-08 — rising lava
- Change: Added a rising lava line that kills the climber once it stops gaining height.
- Run: https://wandb.ai/lealea/climbing_env_mjx/runs/zal186h5

## E18 — 2026-07-08 — 4000 step episode
- Change: Raised the episode cap to 4000 steps.
- Run: https://wandb.ai/lealea/climbing_env_mjx/runs/qji6mxpm

## E19 — 2026-07-08 — 2000 step episode, frameskip 5
- Change: Set a 2000-step episode with frameskip 5.

## E20 — 2026-07-08 — soft lava
- Change: Made lava a soft per-step penalty below the line, terminal only past 0.5 m depth.
https://wandb.ai/lealea/climbing_env_mjx/runs/jptf1flv

## E21 — 2026-07-09 —
- Change: Added more holds to the route.
https://wandb.ai/lealea/climbing_env_mjx/runs/5ivttdfn
https://wandb.ai/lealea/climbing_env_mjx/runs/xzy7esvh

## E22 — 2026-07-09 — more holds and rectangle, long episode
- Change: Switched to rectangular holds with a 4000-step episode cap.
https://wandb.ai/lealea/climbing_env_mjx/runs/0eqx00kq

## E23 — 2026-07-09 — stickier feet
- Change: Made the feet stickier.
https://wandb.ai/lealea/climbing_env_mjx/runs/23lb9wz4

## E24 — 2026-07-10 — rising lava back
- Change: Made the rising lava line terminal again.
https://wandb.ai/lealea/climbing_env_mjx/runs/4lq2kwsz

## E25 — 2026-07-10 — hold only gives reach reward if its above limb
- Change: Paid the reach reward only for holds above the limb.
https://wandb.ai/lealea/climbing_env_mjx/runs/79m47ti7

## E26/1 — 2026-07-10 — frameskip 5 (uniform holds, E21 base)
- Change: Ran E21 with frameskip 5 and uniform holds.

## E26/2 — 2026-07-10 — smaller reach weight
- Change: Cut the reach length scale from 0.5 to 0.3.

## E27 — 2026-07-10 — rising lava but slower
- Change: Slowed the rising lava and made it terminate without a penalty.

## E28 — 2026-07-12 — more holds (E21 base)
- Change: Added more holds to E21, laid out in a zigzag.

## E29 — 2026-07-13 — reach → potential-based shaping
- Change: Converted the reach term from a per-step proximity stream to potential-based shaping.

## E30 — 2026-07-13 — stall truncation (on top of E29)
- Change: Truncated the episode without penalty once best height stalled for 2 s.

## E31 — 2026-07-13 — spawn-height curriculum (on top of E29+E30)
- Change: Added a spawn-height curriculum lifting each reset by up to 2.86 m.

## E32 — 2026-07-13 — lifted spawns start attached (hands in sticky zone)
- Change: Made lifted spawns start with both hands already inside the sticky zone.

## E33 — 2026-07-14 — stickier feet (foot gain on E29 base)
- Change: Raised the foot sticky gain on the E29 base.

## E34 — 2026-07-14 — inclined slab wall (Kang et al.), on E29
- Change: Moved the E29 base onto the Kang et al. inclined slab wall at -15 deg.

## E34' — 2026-07-14 — same configuration, larger budget
- Change: Re-ran E34 unchanged with a 120M-step budget.

## E35 — 2026-07-15 — continue E34. longer ep, real hold
- Change: Continued E34 with a longer episode and a real hold.

## E36 — 2026-07-17 — from laddermen, perlimb movement schedule
- Change: Added a per-limb movement schedule, borrowed from laddermen.

## E38 — 2026-07-18 — E35 with stalling prevention
- Change: Ran E35 with stalling prevention.

## E39 — 2026-07-18 — E35 resume with 2000 step but spheres
- Change: Resumed E35 with 2000-step episodes and sphere holds.

## E40 — 2026-07-18 — E35 resume with 1000 steps and spheres
- Change: Resumed E35 with 1000-step episodes and sphere holds.

## E41 — 2026-07-18 — fingers , gradually less sticky holds (E35)
- Change: Added fingers and rectangular legs, with the holds made gradually less sticky.

## E42 — 2026-08-20 — stronger hands 200 (E34)
- Change: Raised the hand sticky gain to 200 on the E34 base.

## E43 — 2026-08-20 — frameskip 5, typical for gym (E34)
- Change: Ran E34 with frameskip 5 across five seeds.

## E44 — 2026-08-21 — different terms left out
- Change: Left different reward terms out.
