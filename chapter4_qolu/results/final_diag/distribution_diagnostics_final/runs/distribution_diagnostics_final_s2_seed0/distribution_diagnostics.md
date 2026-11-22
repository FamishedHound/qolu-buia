# QOLU return-distribution diagnostics

Generated 2026-10-02T21:36:06+00:00.  Tag `final_s2_seed0`.

## 1. How the distribution is used in the final algorithm

The decision path reads the categorical return distribution `Z(s, a)` **only through its mean**:

* quality score: Q(s, approve) - Q(s, reject), with Q(s, a) = sum_i z_i p_i(s, a) = E[Z(s, a)];
* greedy action: argmax_a E[Z(s, a)];
* consensus: the signed sum of the stakeholders' quality scores, so the consensus is a function of the means alone;
* risk sensitivity: none - no quantile, variance or tail of Z enters a decision anywhere in this package.

Two distributions with the same `E[Z]` and different shape therefore produce the same action, the same quality score and the same consensus decision.  Where the distribution *does* act is the learning rule only - the loss is the KL divergence against the projected n-step target distribution, so the shape of Z shapes the gradients and therefore the mean that is learnt.  Z is an internal model, not a decision variable; the statistics below measure whether that internal model is any good.

## 2. What the distribution was scored against

This environment has no terminal state inside a run, so three readings of the realised return are reported side by side and never mixed:

* `n_step_reward_sum` - bootstrap-free n-step reward sum at the agent's own n and gamma, truncated at the end of the run.
* `n_step_target` - the agent's own training target: the n-step reward sum plus gamma^n times the double-Q bootstrap from its target network.
* `tail_return` **(headline)** - bootstrap-free discounted tail of the appraisal stream, truncated at the end of the run.

Rollout: 200 iterations x 50 appraisals per iteration x 3 stakeholder(s), reward reading `team`, epsilon = 0.00, investor attached: yes.

Target-network schedule: `t_n` = 25 gradient steps; the 3 stakeholder(s) took 980 gradient steps each and synchronised their target networks 39 time(s) each, so the double-Q bootstrap tracked the online network.

## 3. Accuracy and error of the learnt distribution (pooled)

| Statistic | `n_step_reward_sum` | `n_step_target` | `tail_return` |
| --- | --- | --- | --- |
| appraisals scored | 30000 | 30000 | 30000 |
| MAE of E[Z] | 59.6471 | 1.6698 | 90.4261 |
| RMSE of E[Z] | 59.6627 | 2.1093 | 91.1173 |
| bias of E[Z] (predicted - realised) | 59.6471 | -1.6571 | -90.1256 |
| SD of the realised return | 0.4441 | 0.5342 | 13.3449 |
| independent observations (effective n) | 30000 | 30000 | 30000 |
| distinct realised return values | 600 | 29234 | 10000 |
| PIT KS statistic | 0.7735 | 0.5085 | 0.8356 |
| PIT KS p-value | 0.0000 | 0.0000 | 0.0000 |
| coverage of the central 50 % interval (nominal 0.50) | 0.0000 | 1.0000 | 0.0120 |
| mean interval width of the central 50 % interval | 95.9960 | 95.9960 | 95.9960 |
| coverage of the central 90 % interval (nominal 0.90) | 1.0000 | 1.0000 | 0.4522 |
| mean interval width of the central 90 % interval | 208.0000 | 208.0000 | 208.0000 |
| CRPS | 38.4469 | 16.7429 | 55.2273 |
| CRPS skill vs the network at initialisation | -0.1253 | 0.6247 | 0.4046 |
| **CRPS skill vs climatology** | -12.0724 | -7.6344 | -8.5740 |
| boundary-atom mass (atoms 0 and N-1) | 0.0003 | 0.0003 | 0.0003 |
| realised returns outside the support | 0.0000 | 0.0000 | 0.0000 |
| atom spacing delta_z | 8.0000 | 8.0000 | 8.0000 |
| SD of the realised return, in atoms | 0.0555 | 0.0668 | 1.6681 |
| atoms the realised returns land on | 2 | 1 | 23 |
| support resolves the signal | no | no | yes |

Read the support rows column by column.  The three targets live on different scales, and `n_step_target` is the one the network is actually regressed onto, so a support that cannot resolve *that* column is a statement about the algorithm as published - 51 atoms on [-200, 200] against this environment's reward scale - rather than about this particular run.

## 4. Per stakeholder, scored against `tail_return`

| Stakeholder | Objective | MAE of E[Z] | CRPS | CRPS skill vs climatology | PIT KS p | boundary mass |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | land_use | 92.1502 | 56.4400 | -8.7842 | 0.0000 | 0.0004 |
| 1 | woodland | 90.0273 | 55.4434 | -8.6114 | 0.0000 | 0.0002 |
| 2 | urban | 89.1007 | 53.7984 | -8.3263 | 0.0000 | 0.0004 |

## 5. Policy state-dependence

The approval rate alone cannot distinguish a stakeholder that approves everything because it separates nothing from one that approves everything on the merits.  The spread of the quality score across the candidates appraised can: a policy that puts every candidate in the same place has a standard deviation of zero.

| Stakeholder | Objective | approval rate | mean quality score | **SD of the quality score** | SD within an iteration | min | max | SD / mean abs. | discriminates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | land_use | 1.0000 | 0.3638 | 0.0096 | 0.0095 | 0.3415 | 0.3969 | 0.0265 | no |
| 1 | woodland | 1.0000 | 0.7108 | 0.0051 | 0.0051 | 0.6889 | 0.7253 | 0.0072 | no |
| 2 | urban | 1.0000 | 0.9817 | 0.0051 | 0.0050 | 0.9618 | 1.0016 | 0.0052 | no |

Of 3 stakeholder(s), 3 approved either every candidate or none of them, and 0 gave every candidate the same quality score to within 0.001 of its own magnitude.  The quality-score standard deviation runs from 0.005121 to 0.00963 across stakeholders: no stakeholder both varied that score and split its verdicts, so the policy separates no candidate from another and its approval rate is a property of the network rather than of the states it appraised.

A stakeholder counts as *discriminating* when its quality-score spread exceeds 0.001 of the score's own mean magnitude **and** its approval rate lies strictly between 0 and 1.  The threshold is a reporting convenience; the raw spread is in the table so a reader can apply their own.

## 6. Verdict

CRPS skill against climatology is -8.574: the return distribution carries no more information than the run's marginal histogram, i.e. no state-conditional internal model is evident. The randomised PIT departs from uniformity (KS = 0.836, p = 0), so the distribution is not calibrated. Mass on the two extreme atoms averages 0.0003, so the support holds the distribution. The atom spacing is delta_z = 8 against a realised return of standard deviation 13.3 (1.67 atoms), and the run's returns occupy 23 of 51 atoms, so the support can express the variation being predicted. The mean absolute error of E[Z], the quantity that actually drives decisions, is 90.4261 against a realised return with standard deviation 13.3449. 30000 appraisals were scored over 200 iterations and take 10000 distinct realised values, so the nominal count overstates the independent information behind every statistic above.

The KS p-value is reported as a descriptive threshold, not as a significance test: an evaluation rollout's appraisals are not independent (50 share a state within an iteration, and consecutive iterations share a landscape), so the effective sample size is far below the nominal one and a p-value computed as if the sample were i.i.d. is anti-conservative at these sizes.  Read the KS statistic, which is an effect size, rather than the p-value beside it.

| Question | Answer |
| --- | --- |
| Is `Z` state-conditional (positive **CRPS skill** vs climatology)? | no |
| Is `Z` calibrated (randomised PIT consistent with uniform)? | no |
| Is the atom support adequate (low **boundary**-atom mass)? | yes |
| Is the atom spacing fine enough to express the signal? | yes |
| Does the policy depend on anything but `E[Z]`? | no |

The CRPS skill against climatology is the number that answers the first question of the table above: climatology always emits the run's own marginal histogram of realised returns, so it uses no state information at all, and beating it is exactly the claim that a state-conditional internal model has been learnt.

