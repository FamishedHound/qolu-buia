# Baseline specifications: LToS and MitM

Background note on the two baselines of `chapter4_qolu` whose implementation
the IJCAI-2025 paper leaves open.

**Target environment (from the published paper).** 50x50 grid of 1 km2 parcels
(UKCEH aggregate classes); per iteration 50 candidate parcels are drawn with
`P(sampled | A_i) ~ A_i` (Eq. 2); agents convert a bounded number of them to
urban; three stakeholder objectives (distance from bog/mountain/heath;
preservation of woodland + arable; proximity to urban/suburban) are mixed under
four weight scenarios (Table 3). Source:
[IJCAI-25 proceedings PDF](https://www.ijcai.org/proceedings/2025/0027.pdf).
Note: the published text says "50 parcels are sampled with replacement, allowing
for the potential conversion of up to 11 land types into urban areas" - it does
**not** state a per-iteration conversion budget of 10 parcels. Treat the budget
as an implementation parameter, not a published fact. **[ASSUMPTION]**

The paper names the baselines (Section 4, "Baselines") but gives **no**
implementation detail for either, so both are implemented from their source
publications.

---

## 1. LToS - Learning to Share in Multi-Agent Reinforcement Learning

Yi, Li, Wang & Lu. arXiv:2112.08702 (2021); NeurIPS 2022.
[arXiv abs](https://arxiv.org/abs/2112.08702) |
[arXiv PDF (contains Appendices A-D)](https://arxiv.org/pdf/2112.08702) |
[NeurIPS camera-ready](https://proceedings.neurips.cc/paper_files/paper/2022/file/61d8577984e4ef0cba20966eb3ef2ed8-Paper-Conference.pdf) |
[OpenReview](https://openreview.net/forum?id=awnQ2qTLSwn)

Everything in 1.1-1.5 is **VERIFIED** against the arXiv PDF (full text and
appendices).

### 1.1 Problem setting
Networked MARL. Agents `V = {1..N}` sit on an undirected graph `G(V,E)`; agent
`i` communicates only with `N_i` (its neighbours, with a self-loop `d_ii`
included). Assumption inherited from Qu et al. 2020a: `r_i(s,a) = r_i(s, a_{N_i})`.
Partial observability: `o_i` stands in for `s` (observations of `N_i`, or
history). Global objective: maximise `sum_t gamma^t sum_i r_i^t`.

### 1.2 Agents / observations / actions
Two policies per agent, hierarchically decentralised:
- **High level** `phi_i : o_i -> w_i^out`, a continuous vector
  `w_i^out = {w_ij | j in N_i}` with `w_ij in [0,1]` and `sum_{j in N_i} w_ij = 1`
  (enforced by a softmax output layer). Learned with DDPG.
- **Low level** `pi_i : (o_i, w_i^in) -> a_i`, where `w_i^in = {w_ji | j in N_i}`
  are the weights *incoming* from neighbours. Learned with DGN (graph-convolutional
  Q-learning, Jiang et al. ICLR 2020).

### 1.3 Objective and update rules
Shaped reward (Eq. 2 of the paper):

```
r_i^w = sum_{j in N_i} w_ji * r_j
```

Because each row sums to 1, `sum_i r_i^w = sum_i r_i`: **reward sharing is
conservative, so the global objective is unbiased**. This is precisely why it is
a legitimate baseline against a team-reward method.

Bi-level optimisation (Eq. 4):

```
max_phi  J_phi( pi*(phi) )    s.t.   pi*(phi) = arg lim_pi J_pi(phi)
```

Solved with first-order approximations; second-order terms are dropped (as in
MAML / DARTS). Key identity used for the implementation:

```
q^{phi_i}(o_i, w_i^out) = v^{pi_i}(o_i ; w_i^in)
```

i.e. the DGN Q-network doubles as the DDPG critic; the high-level gradient is
taken w.r.t. `w_i^in` at the argmax low-level action.

### 1.4 Architecture and published hyperparameters
Low level = DGN: observation encoder MLP -> multi-head graph attention over
`N_i` -> Q head. High level = DDPG actor with softmax output over `|N_i|`.

Traffic 6x6 column (closest published analogue to a fixed 2-D lattice of spatial
agents), from Tables 4 and 5:

| Low level (DGN) | value | High level (DDPG) | value |
|---|---|---|---|
| sample size / batch | 1000 / 20 | sample size / batch | 1000 / 20 |
| buffer capacity | 10,000 | update frequency | every 20 episodes |
| eps start/decay/end | 0.4 / 0.9 / 0.05 | action interval `M` | 15 steps |
| optimiser, lr | RMSProp, 1e-3 | optimiser, lr | SGD, 1e-3 |
| gamma | 0.8 | exploration noise | Ornstein-Uhlenbeck, sigma = 0.25 |
| tau (soft update) | 0.1 | initialiser | "selfishness" initialiser, 0.8 |
| MLP units / activation | 32 & 32, ReLU | last-layer activation | softmax |

Jungle column (30x30 grid game, 20 agents, dynamic topology - closest analogue
to *moving* candidate-parcel agents): Adam 1e-4, gamma 0.96, tau 0.01, MLP
512 & 128, buffer 200k, batch 10, eps 0.6/0.996/0.01; high level SGD 1e-4,
OU sigma 0.025, batch 32, update every 100 episodes, initial selfishness 0.5.
*(The attention-head and encoder-layer counts are not reproduced here; see
Table 4 of the PDF. This package uses standard DGN defaults for them - see
`LToSConfig` in `baselines/config.py`.)*

Three training tricks from Appendix C, all needed for stability:
1. **Selfishness initialiser** - bias of the last FC layer set so the agent
   initially keeps a chosen self-weight (0.8 traffic 6x6, 0.9 Shenzhen, 0.5
   prisoner/jungle), remainder split evenly over neighbours.
2. **Unified PRNG / shared seed** so all agents sample the *same timesteps* from
   their independent replay buffers (required because `w_i^in` must be consistent
   across the batch).
3. **Different time scales** - the high level acts every `M` steps (15 in traffic).

### 1.5 Training loop (Algorithm 1, condensed)
```
for t in 0..T:
  for each agent i:
     exchange observations -> o_i
     w_i^out <- phi_i(o_i) + exploration noise
     exchange w^out -> w_i^in
     a_i <- pi_i(o_i ; w_i^in) with eps-greedy
     execute a_i, observe r_i, o_i'
     exchange r -> r_i^w = sum_{j in N_i} w_ji r_j
     store (o_i, w_i^in, a_i, r_i^w, o_i', N_i) in B_i
  if t mod update_frequency == 0:
     for each agent i:
        sample synchronised minibatch D from B_i
        y_i <- r_i^w + gamma * q'_i(o_i', a'; w_i^in')          # target nets
        update critic mu_i on (y_i - q_i(o_i, a_i; w_i^in))^2
        g_i^in <- grad_{w_i^in} q_i(o_i, argmax_a q_i ; w_i^in)
        exchange g^in -> g_i^out
        update theta_i by (1/|D|) sum_o (grad_theta phi_i(o_i))^T g_i^out
        soft-update target networks
```

### 1.6 Mapping LToS onto the land-use environment
LToS needs (a) many agents, (b) a *per-agent environment reward*, (c) a sparse
neighbour graph, (d) global objective = sum of local rewards. The published
environment supplies a *shared team reward*, so a decomposition must be invented.
Three candidate mappings, with the assumptions each forces:

**(A) District lattice - recommended.** Tile the 50x50 parcel grid into `K`
districts (e.g. 10x10 = 100 districts of 5x5 parcels); one LToS agent per
district; `E` = 4-neighbour (rook) adjacency.
- `o_i`: local land-cover one-hot planes + attractiveness patch for the district
  (reuse the windowing logic in `env/urban_env.py::buia_observations`).
- `a_i`: per candidate parcel inside district `i`, convert / do-not-convert
  (discrete, as DGN requires).
- `r_i`: the **published team reward restricted to conversions inside district i**.
  Each of the three objective terms in `env/rewards.py`
  (`land_use_separation_reward`, `woodland_preservation_reward`,
  `urban_proximity_reward`) is computed per converted parcel, so attribution by
  containing district is exact and `sum_i r_i` equals the global team reward.
  **[ASSUMPTION: global normalisers (e.g. max-distance denominators) must stay
  whole-map constants, not per-district, or the sum stops matching the published
  objective.]**
- "Sharing" = a district learning how much of its own objective satisfaction to
  credit to neighbouring districts - a direct analogue of planning-authority
  spillover.
- **[ASSUMPTION]** the per-iteration conversion budget is a *global* constraint
  LToS has no mechanism for. Either (i) let districts act freely and enforce the
  budget by keeping the top-B proposals ranked by Q-value (breaks strict
  decentralisation), or (ii) give each district a quota (changes the problem).
  `baselines/ltos.py` takes option (i).

**(B) One agent per stakeholder objective (N = 3, complete graph).** "Sharing"
becomes a learned redistribution of reward across the three objectives. Honest,
but weak: with N = 3 and a complete graph there is no network structure, and the
learned weights compete directly with the scenario weights of Table 3 - a
confound that makes the four-scenario comparison uninterpretable.
**Not recommended.**

**(C) Candidate-parcel agents (jungle-like).** The 50 sampled candidates each
iteration are the agents; `N_i` = `k` nearest candidates by Euclidean distance;
`a_i` in {convert, skip}. Closest to the published environment loop and to LToS's
dynamic-topology jungle experiment. **[ASSUMPTION: agent identity does not
persist across iterations, so all agents must share one parameter set. LToS's
jungle keeps persistent agents and varies only topology; parameter sharing is an
extension, not published behaviour.]**

**Assumptions common to all mappings.** (i) The society / land-preference term
`P_society` (Eq. 3) is not a stakeholder, so it must either be added to every
local reward or excluded from the decomposition; `baselines/ltos_rewards.py`
excludes it. (ii) The
environment's 1000 iterations map to one episode, so the traffic `gamma = 0.8`
is probably too aggressive; prefer the jungle value 0.96 or the value already in
`chapter4_qolu/config.py`. (iii) LToS is single-objective: it only ever sees the
scalarised team reward, so it must be retrained once per scenario.

---

## 2. MitM - Modeling-in-the-middle (Bone, Dragicevic & White, 2011)

*IJGIS* 25(5):717-737. DOI
[10.1080/13658816.2010.495076](https://doi.org/10.1080/13658816.2010.495076).

### 2.1 Access status
**Paywalled.** The paper is not open access, and neither are the two
methodological companion papers: Bone & Dragicevic, *Transactions in GIS*
13(3):253-272, 2009
([DOI](https://doi.org/10.1111/j.1467-9671.2009.01151.x)) and *Computers,
Environment and Urban Systems* 34(2):162-174, 2010
([DOI](https://doi.org/10.1016/j.compenvurbsys.2009.10.001)). No figures,
equations or hyperparameters from their full texts are used. *Everything from
2.3 onward is inferred.*

### 2.2 VERIFIED content
Full abstract, retrieved from the
[CoLab DOI record](https://colab.ws/articles/10.1080%2F13658816.2010.495076):
agents represent "households and commercial enterprises" that "select locations
to inhabit based on population densities and attractivity preferences"; the
resulting land-use change "is evaluated by a set of agents representing different
stakeholders who are embedded with RL algorithms that allow them to influence the
land use change process so that their objectives are addressed"; the outcome is
"negotiated land use patterns in which the desires and objectives of all
individuals are constrained by behaviors of others".

The 45-item reference list (verified via the
[OpenAlex record](https://api.openalex.org/works/doi:10.1080/13658816.2010.495076))
constrains the method strongly:
- RL family: **Sutton 1988 (TD learning)**, **Barto & Sutton 1981 (associative
  search network)**, Russell & Norvig. **No Watkins / Q-learning citation appears**
  - so the learner is very likely TD(0)-style value estimation or associative
  reward-driven weight search, *not* Q-learning. **[INFERRED]**
- MODM family: **Cohon 1978, *Multiobjective Programming and Planning***;
  **Eastman & Jiang 1998, multi-criteria / multi-objective land allocation (MOLA)**;
  Malczewski GIS-MCDA surveys; Xiao & Bennett evolutionary multiobjective site
  search; Ligmann-Zielinska & Church spatial optimisation.
- Land-use dynamics: **White & Engelen 1997 (constrained CA, "attractivity")**,
  Benenson 1998 (multi-agent residential dynamics), Clarke, Batty, Engelen MOLAND.

The 2009 companion abstract (verified via OpenAlex) describes the same machinery
as an "Intelligent Agent Model": ABM inside a GIS plus RL, agents = forest
companies maximising economic return while minimising adverse ecological impact,
with validation against practical decision-making solutions.

### 2.3 Inferred method - ALL INFERRED
Three nested layers, iterated:

1. **Attractivity surface (White & Engelen form).** For cell `c`, land use `k`:
   `A_k(c) = s_k(c) * ( 1 + sum_d sum_k' w_{k,k',d} * n_{k',d}(c) ) * (1 + eps)`
   where `s_k` is intrinsic suitability, `n_{k',d}` counts cells of class `k'` in
   distance ring `d`, and `eps` is a stochastic perturbation.
2. **Bottom-up ABM.** Household / enterprise agents pick a location by sampling
   or arg-maxing `U(c) = A_k(c) * f(population density)`; totals driven by
   exogenous demand.
3. **Top-down stakeholder RL.** Stakeholder `m` holds a criterion-weight vector
   `theta_m` over the MCE layers that build `s_k` and `w_{k,k',d}`. After each
   round it scores the realised pattern against its objective, `R_m =
   O_m(pattern)`, and updates by a TD / associative rule
   `V_m <- V_m + alpha * (R_m - V_m)`, nudging `theta_m` toward settings whose
   realised reward exceeded `V_m` (softmax or hill-climb over candidate `theta_m`).
4. **The "middle".** Revised `theta_m` re-shapes the attractivity surface
   (protected-area masks, incentives) that bottom-up agents respond to next round;
   MODM aggregation (Cohon compromise programming / Eastman-Jiang MOLA) resolves
   conflicts. Convergence = a stable "negotiated" pattern.

```
for round in 1..T:
    A <- attractivity(land_cover, theta_1..theta_M)
    proposals  <- bottom_up_agents_choose(A, demand)          # myopic, no learning
    allocation <- MODM_resolve(proposals, theta_1..theta_M)   # MOLA / compromise
    apply(allocation); update land_cover, A
    for m in stakeholders:
        R_m     <- objective_m(land_cover_before, land_cover_after)
        theta_m <- nudge(theta_m, R_m - V_m)
        V_m     <- V_m + alpha * (R_m - V_m)
```

### 2.4 Mapping MitM onto the land-use environment
This mapping is unusually clean because the IJCAI environment was **explicitly
designed after MitM**: the abstract says the approach is "inspired by the work of
Bone et al. [2011]", and Related Work says "early attempts at
'modeling-in-the-middle' [Bone et al., 2011] underscored the need for linking
policy instruments".

| MitM element | Environment element | Status |
|---|---|---|
| attractivity surface | `B_attractiveness` / `env/attractiveness.py` (Eq. 1) | verified analogue |
| households/enterprises picking by attractivity | `P(sampled) ~ A_i` (Eq. 2) plus a *non-learning* greedy chooser | **[ASSUMPTION]** the bottom-up layer must stay a myopic utility maximiser; using the neural BUIA would turn MitM into QOLU |
| stakeholder agents with RL | the three `TopDownObjective` terms in `objectives.py` | verified analogue |
| stakeholder reward `O_m` | `land_use_separation_reward`, `woodland_preservation_reward`, `urban_proximity_reward` | **[ASSUMPTION]** MitM's originals were MODM objective-function values, not these reward terms |
| MODM aggregation | scenario weights, Table 3 | **[ASSUMPTION]** MitM negotiates weights endogenously; here they are exogenous. Either freeze `theta` at the scenario weights (weaker, comparable) or let MitM learn them (truer to MitM, but the scenario axis then loses meaning). This package freezes `theta` by default (`mitm`) and offers the learned variant (`mitm_learned`). |
| exogenous demand | population influx / `well_being.py` (`H_occupied / H_total`) | **[ASSUMPTION]** |
| conversion budget | bounded conversions per iteration | **[ASSUMPTION]** MitM has no budget; enforce inside the MODM step by taking the top-B by compromise score |
| stakeholder influence channel | modulating attractiveness weights **or** vetoing candidates | **[UNVERIFIED]** the actual mechanism is unknown; the abstract only says the agents "influence the land use change process" |

**Design note.** MitM is a 2011 GIS/ABM method, not a deep-RL
method. Implementing it with a tabular / linear TD learner over criterion weights
is *faithful*; upgrading it to a deep network would silently strengthen the
baseline and misrepresent the comparison. The implementation's assumptions are
stated in `baselines/mitm.py` and `baselines/mitm_allocation.py`.

---

## 3. Open-source implementations found

| Repo | Lang | Licence | Assessment |
|---|---|---|---|
| [PKU-RL/RoadnetSZ](https://github.com/PKU-RL/RoadnetSZ) (`code/LToS`) | Python (TF1/Keras) | **none declared** (all rights reserved) | The *official* LToS code, cited in the paper as `PKU-AI-Edge/RoadnetSZ`; a fork of CoLight, entirely traffic-specific (CityFlow/SUMO, `anon_env.py`, `Swarm_CoLight_agent.py`). Read it for the sharing and gradient-exchange mechanics (`check_w_in.py`, `noise.py`, `updater.py`); do not port wholesale. |
| [PKU-RL/DGN](https://github.com/PKU-RL/DGN) | Python | none declared | Official low-level policy of LToS; the `Jungle/` grid-world variant is the closest published template for parcel-lattice observations plus a graph-attention Q head. |
| [wingsweihua/colight](https://github.com/wingsweihua/colight) | Python | none declared | Upstream of the LToS fork; useful mainly to diff against it and isolate the LToS-specific additions. |
| [sunoh-kim/graph-convolutional-reinforcement-learning](https://github.com/sunoh-kim/graph-convolutional-reinforcement-learning) | Python | none declared | Third-party DGN reproduction (Routing scenario); small and readable, a sanity reference for the attention block. |
| [michstrauch/CoMOLA](https://github.com/michstrauch/CoMOLA) | Python | GPL-3.0 | Constrained multi-objective land-use allocation by genetic algorithm; the best open reference for MitM's MODM/allocation layer. GPL-3.0 is copyleft - use as reference, do not vendor. |
| [gracemilner/MOLA_exercise](https://github.com/gracemilner/MOLA_exercise) | Python | none declared | Tiny worked MOLA (Eastman & Jiang) example; good for checking the compromise-programming arithmetic. |
| [mesa/mesa-geo](https://github.com/mesa/mesa-geo) | Python | Apache-2.0 | Maintained GIS extension for Mesa; a viable host if the MitM baseline is written as a standalone ABM rather than inside `chapter4_qolu/env`. |
| [nextgis/qgis_molusce](https://github.com/nextgis/qgis_molusce) | Python | GPL-2.0 | Land-use-change simulation modules (QGIS plugin); reference for transition-potential surfaces, not for RL. |

**No open-source implementation of MitM exists.** None was found on GitHub, and
the paper predates code-release norms in GIScience.

---

## 4. Choices made in this package
1. **LToS:** mapping (A) on a 10x10 district lattice, reusing the per-parcel
   terms in `env/rewards.py` for the reward decomposition; the traffic-6x6
   hyperparameters, except `gamma`, which is taken from
   `chapter4_qolu/config.py` rather than 0.8; the selfishness initialiser and
   the shared-seed replay sampling - Appendix C says both are load-bearing.
2. A **`fixed LToS` ablation** (weights frozen at the initialiser). The
   original paper reports it in all three scenarios and it is nearly free.
3. **MitM:** a non-neural agent - attractivity surface, greedy bottom-up
   chooser, per-stakeholder TD(0) weight learner, and compromise-programming
   allocation of the top-B candidates - with `theta` frozen at the Table 3
   scenario weights for the headline comparison and the weight-learning variant
   as a secondary row.
4. Every `[ASSUMPTION]` above is recorded as a `Design note` where the code
   makes it. The baseline numbers in Tables 4/5 of the IJCAI paper cannot be
   reproduced exactly from the paper, which does not specify the baselines.

## 5. Sources
1. [Yi et al., *Learning to Share in Multi-Agent RL*, arXiv:2112.08702](https://arxiv.org/abs/2112.08702) - primary LToS source; appendix read in full.
2. [NeurIPS 2022 camera-ready of LToS](https://proceedings.neurips.cc/paper_files/paper/2022/file/61d8577984e4ef0cba20966eb3ef2ed8-Paper-Conference.pdf) - main text, no appendix.
3. [Bone, Dragicevic & White 2011, DOI record](https://doi.org/10.1080/13658816.2010.495076) - paywalled.
4. [CoLab record with the full MitM abstract](https://colab.ws/articles/10.1080%2F13658816.2010.495076) - only verified textual source for MitM.
5. [OpenAlex record for MitM (reference list)](https://api.openalex.org/works/doi:10.1080/13658816.2010.495076) - used to infer the RL and MODM families.
6. [Bone & Dragicevic 2009, *Transactions in GIS*](https://doi.org/10.1111/j.1467-9671.2009.01151.x) - companion "Intelligent Agent Model"; abstract only.
7. [Bone & Dragicevic 2010, *CEUS*](https://doi.org/10.1016/j.compenvurbsys.2009.10.001) - RL ABM for forest management; abstract only.
8. [Pelcner et al., IJCAI-25 proceedings PDF](https://www.ijcai.org/proceedings/2025/0027.pdf) - environment, metrics, scenarios, and the baseline list.
9. [IJCAI-25 paper landing page](https://www.ijcai.org/proceedings/2025/27) - DOI 10.24963/ijcai.2025/27.
10. [Jiang et al., *Graph Convolutional RL* (DGN) code](https://github.com/PKU-RL/DGN) - LToS low-level policy.

## 6. Open questions
- What exactly did MitM's stakeholder agents *act on* - criterion weights, vetoes,
  or zoning masks? Requires the paywalled PDF.
- MitM's learning rate, number of stakeholders, grid size, and number of rounds:
  all unknown.
- Which LToS mapping the IJCAI experiments used is not stated; the Baselines
  section gives one sentence per baseline.
