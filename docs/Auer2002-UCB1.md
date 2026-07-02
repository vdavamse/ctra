# Finite-time Analysis of the Multiarmed Bandit Problem

## Authors and Affiliations

- **Peter Auer**, University of Technology Graz, A-8010 Graz, Austria (pauer@igi.tu-graz.ac.at)
- **Nicolo Cesa-Bianchi**, DTI, University of Milan, I-26013 Crema, Italy (cesa-bianchi@dti.unimi.it)
- **Paul Fischer**, Lehrstuhl Informatik II, Universitat Dortmund, D-44221 Dortmund, Germany (fischer@ls2.informatik.uni-dortmund.de)

## Venue/Journal

- **Journal**: Machine Learning, 47, 235-256, 2002
- **Publisher**: Kluwer Academic Publishers
- **Editor**: Jyrki Kivinen
- **DOI**: 10.1023/A:1013689704352
- **Note**: Preliminary version appeared in Proc. of 15th ICML, pages 100-108, Morgan Kaufmann, 1998

## Abstract

Reinforcement learning policies face the exploration versus exploitation dilemma, i.e., the search for a balance between exploring the environment to find profitable actions while taking the empirically best action as often as possible. A popular measure of a policy's success in addressing this dilemma is the regret, the loss due to not following the globally optimal policy at all times. One of the simplest examples is the multi-armed bandit problem. Lai and Robbins were the first to show that the regret for this problem has to grow at least logarithmically in the number of plays. Since then, policies which asymptotically achieve this regret have been devised. In this work, the authors show that the optimal logarithmic regret is also achievable uniformly over time, with simple and efficient policies, and for all reward distributions with bounded support.

## Problem Statement

1. **Exploration-exploitation tradeoff**: In the K-armed bandit problem, an agent must choose between exploring to discover which arm has the highest reward and exploiting the currently best-known arm.
2. **Finite-time guarantees needed**: Prior work (Lai and Robbins, 1985) proved asymptotic logarithmic regret bounds but did not provide finite-time guarantees — existing policies could have arbitrarily poor regret for finite horizons.
3. **Computational complexity**: Lai and Robbins' policies require computing complex upper confidence indices based on entire reward sequences, making them computationally expensive.

## Key Contributions

1. **UCB1 algorithm**: A simple, deterministic policy achieving O(log n) regret uniformly over time for all reward distributions with bounded support [0,1]. The policy selects the arm maximizing: x̄_i + sqrt(2 ln n / n_i), where x̄_i is the sample mean, n is total plays, and n_i is plays of arm i.
2. **UCB2 algorithm**: A more complex epoch-based variant where the leading constant in the regret bound can be made arbitrarily close to the information-theoretic lower bound 1/(2Δ²_i).
3. **ε_n-GREEDY**: A logarithmic-regret variant of the classical ε-greedy heuristic with ε decaying as 1/n.
4. **UCB1-NORMAL**: An extension for normally distributed rewards with unknown means and variances.
5. **Finite-time regret bounds**: All bounds hold uniformly over time (not just asymptotically), strengthening prior results.

## Methodology/Architecture

### UCB1 Policy

1. Play each arm once (initialization)
2. At each subsequent step, play arm j that maximizes:
   ```
   x̄_j + sqrt(2 ln n / n_j)
   ```
   - First term (x̄_j): exploitation — sample mean reward
   - Second term: exploration — upper confidence bound derived from Chernoff-Hoeffding inequality

### Regret Bound (Theorem 1)

For K arms with reward distributions supported on [0,1]:
```
E[regret after n plays] ≤ Σ_{i: μ_i < μ*} (8 ln n / Δ_i) + (1 + π²/3) Σ_{j=1}^{K} Δ_j
```
where Δ_i = μ* - μ_i is the gap between optimal and arm i.

### UCB2 Policy

Uses epochs of exponentially increasing length τ(r) = ⌈(1+α)^r⌉. In each epoch, plays the arm maximizing x̄_i + a_{n,r_i} where the confidence term involves the epoch structure. By choosing α small, the leading constant approaches 1/(2Δ²_i), close to the information-theoretic lower bound.

### ε_n-GREEDY Policy

With probability (1 - ε_n): play the arm with highest sample mean. With probability ε_n: play a random arm. Setting ε_n = min(1, cK/(d²n)) where d ≤ min Δ_i achieves logarithmic regret.

## Datasets

- **Theoretical analysis**: Proofs apply to all K-armed bandit instances with reward distributions supported on [0,1]
- **Experimental validation**: Synthetic bandit problems with Bernoulli rewards
  - 10-armed bandit: one arm with mean 0.9, others with mean 0.8 (small gap Δ=0.1)
  - 10-armed bandit: one arm with mean 0.55, others with mean 0.45 (larger gap Δ=0.1 but lower absolute rewards)
  - Horizon: up to 100,000 plays

## Results

- **UCB1**: Achieves logarithmic regret with leading constant 8/Δ²_i, which is a factor of 16 worse than the information-theoretic lower bound 1/(2Δ²_i) but with the advantage of simplicity and no distributional assumptions.
- **UCB2**: Achieves leading constant arbitrarily close to 1/(2Δ²_i) by tuning α → 0, at the cost of a larger additive constant c_α.
- **ε_n-GREEDY**: Achieves logarithmic regret but with a leading constant of (c/d² + 1/K) · Σ 1/Δ_j, generally larger than UCB1.
- **Experimental comparison**: UCB1 outperforms ε_n-GREEDY in practice. UCB2 can be competitive with UCB1 but is sensitive to the α parameter.
- All results hold uniformly over time (not just asymptotically).

## Limitations

1. **Constant factor gap**: UCB1's leading constant 8/Δ²_i is 16x worse than the information-theoretic optimum. UCB2 closes this gap but at the cost of complexity and parameter sensitivity.
2. **Known reward range required**: UCB1 assumes rewards are bounded in [0,1]. Extensions to unknown or unbounded distributions require different confidence bounds.
3. **Independent arms assumption**: The analysis assumes independence between arms, which may not hold in structured problems like tree search.
4. **No contextual information**: UCB1 does not incorporate side information about arms, limiting its direct applicability to contextual settings.

## Relevance to CTRA

This paper provides the mathematical foundation for CTRA's MCTS tree policy, referenced in `README.md` as ref [30]. Key connections:

1. **UCB1 in UCT**: The UCT (Upper Confidence Bounds for Trees) algorithm used in CTRA's MCTS directly applies UCB1 to tree search. Each tree node treats its children as bandit arms, using UCB1's exploration-exploitation formula to select which branch to expand. The formula `Q(s,a) + C * sqrt(ln N(s) / N(s,a))` in CTRA's `src/ctra/search/mcts.py` is a direct instantiation of UCB1.

2. **Logarithmic regret guarantee**: UCB1's finite-time logarithmic regret bound provides the theoretical guarantee that CTRA's MCTS will converge — the algorithm will explore suboptimal feature engineering branches logarithmically often while focusing on the most promising ones.

3. **Exploration constant C**: The paper's analysis of the confidence bound term informs the choice of the exploration constant in CTRA's UCT formula. The UCT exploration paper (ref [31]) builds directly on this work to propose adaptive strategies for C.

## Code/Data Availability

- **Code**: Not applicable (theoretical paper from 2002)
- **Algorithms**: UCB1 is simple enough to implement directly from the paper's pseudocode (Figure 1)
- **Implementations**: Available in virtually all MCTS libraries

## Citation

```bibtex
@article{auer2002finite,
  title={Finite-time Analysis of the Multiarmed Bandit Problem},
  author={Auer, Peter and Cesa-Bianchi, Nicol{\`o} and Fischer, Paul},
  journal={Machine Learning},
  volume={47},
  pages={235--256},
  year={2002},
  publisher={Kluwer Academic Publishers},
  doi={10.1023/A:1013689704352}
}
```
