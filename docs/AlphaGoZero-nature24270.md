# Mastering the Game of Go without Human Knowledge

## Authors and Affiliations

- **David Silver**\*, DeepMind, London, UK
- **Julian Schrittwieser**\*, DeepMind
- **Karen Simonyan**\*, DeepMind
- **Ioannis Antonoglou**, DeepMind
- **Aja Huang**, DeepMind
- **Arthur Guez**, DeepMind
- **Thomas Hubert**, DeepMind
- **Lucas Baker**, DeepMind
- **Matthew Lai**, DeepMind
- **Adrian Bolton**, DeepMind
- **Yutian Chen**, DeepMind
- **Timothy Lillicrap**, DeepMind
- **Fan Hui**, DeepMind
- **Laurent Sifre**, DeepMind
- **George van den Driessche**, DeepMind
- **Thore Graepel**, DeepMind
- **Demis Hassabis**, DeepMind

\* Equal contribution

## Venue/Journal

- **Journal**: Nature, Volume 550, Pages 354-359
- **Date**: 19 October 2017
- **DOI**: 10.1038/nature24270

## Abstract

A long-standing goal of artificial intelligence is an algorithm that learns, tabula rasa, superhuman proficiency in challenging domains. AlphaGo Zero achieves superhuman performance in the game of Go by training solely through self-play reinforcement learning, starting from random play with no human data, no domain knowledge beyond the game rules. AlphaGo Zero uses a single neural network that is trained to predict both AlphaGo's own move selections and the winner of AlphaGo's games. This neural network improves the strength of tree search, resulting in higher quality move selection and stronger self-play in the next iteration. Starting tabula rasa, the new program surpassed the strength of AlphaGo Lee (which defeated world champion Lee Sedol) in just 36 hours, accumulated over 14 million games of self-play, and achieved superhuman performance using a single machine with 4 TPUs.

## Problem Statement

1. **Dependence on human expert data**: Previous approaches (including AlphaGo Fan/Lee) relied on supervised learning from human expert games as a starting point, imposing a ceiling on performance bounded by human knowledge.
2. **Separate policy and value networks**: AlphaGo Lee used separate neural networks for move prediction and position evaluation, preventing shared representation learning and requiring more computation.
3. **Handcrafted features**: Earlier Go programs used domain-specific features designed by human experts, limiting generalization.

## Key Contributions

1. **Tabula rasa learning**: Demonstrates that superhuman Go play can be achieved with zero human knowledge, using only self-play reinforcement learning from random initialization.
2. **Unified dual-head network**: A single neural network f_theta(s) = (p, v) outputs both move probabilities p and position evaluation v, enabling shared representation learning. The dual objective regularizes the network.
3. **Simplified MCTS**: Removes Monte Carlo rollouts entirely, using only the neural network's value output for position evaluation (no fast rollout policy needed).
4. **Residual network architecture**: Uses a deep residual network (20 or 40 blocks) with batch normalization, outperforming the convolutional architecture of AlphaGo Lee by >600 Elo.
5. **Training efficiency**: Surpasses AlphaGo Lee (100-0) in just 72 hours on a single machine with 4 TPUs, compared to months of distributed training for AlphaGo Lee.

## Methodology/Architecture

### Self-Play Reinforcement Learning Pipeline

1. **Self-play data generation**: At each iteration i, games are played using MCTS guided by the previous network f_{theta_{i-1}}. Each position stores (s_t, pi_t, z_t) where pi_t is the MCTS search probability and z_t = ±r_T is the game outcome.
2. **Neural network training**: Parameters theta are updated to minimize:
   ```
   l = (z - v)^2 - pi^T log(p) + c||theta||^2
   ```
   (value loss + policy cross-entropy + L2 regularization)
3. **Iteration**: Improved network produces better self-play data, which trains an even better network.

### MCTS with Neural Network Guidance

Each simulation traverses the tree selecting moves that maximize:
```
Q(s,a) + U(s,a), where U(s,a) ∝ P(s,a) / (1 + N(s,a))
```
- **P(s,a)**: Prior probability from neural network's policy head
- **N(s,a)**: Visit count
- **Q(s,a)**: Mean value from simulations passing through this edge

At leaf nodes, the network evaluates the position: (P(s',·), V(s')) = f_theta(s'). No rollouts needed.

Move selection uses: pi_a ∝ N(s,a)^{1/tau} where tau is a temperature parameter.

### Neural Network Architecture

- **Input**: 19x19x17 binary feature planes (stone positions for last 8 moves + color)
- **Body**: 20 or 40 residual blocks, each with two 256-filter 3x3 convolutional layers + batch normalization + ReLU + skip connection
- **Policy head**: 1x1 conv → FC → softmax over 19x19+1 moves
- **Value head**: 1x1 conv → FC(256) → FC(1) → tanh ∈ [-1, 1]

## Datasets

- **Training data**: Self-generated through self-play (no external datasets)
  - 4.9 million games of self-play generated during 72 hours of training
  - Each game produces ~200 position-move-outcome triples
  - 700,000 mini-batches of 2,048 positions for training
- **Evaluation**: AlphaGo Lee match (100 games), AlphaGo Master match (100 games), internal Elo rating tournament
- **Comparison**: KGS Server dataset (human expert games) used only for supervised learning baseline comparison

## Results

- **AlphaGo Zero vs AlphaGo Lee**: 100-0 after 72 hours of training (single machine, 4 TPUs vs distributed 48 TPUs)
- **AlphaGo Zero surpasses AlphaGo Lee** after just 36 hours of self-play training
- **AlphaGo Zero (40 blocks) vs AlphaGo Master**: 89-11 (Master previously defeated 60 top professionals online)
- **Architecture comparison**: Residual network > convolutional network by >600 Elo; combined policy+value > separate networks by ~600 Elo
- **Supervised learning comparison**: Self-play RL achieves worse move prediction accuracy but much stronger play than supervised learning, suggesting AlphaGo Zero learns qualitatively different strategies from human play
- **Knowledge discovery**: AlphaGo Zero independently rediscovered common Go joseki (corner sequences) during training, then moved beyond them to discover novel strategies

## Limitations

1. **Perfect information assumption**: Designed for fully observable, deterministic, two-player zero-sum games. Does not directly extend to partially observable or stochastic environments.
2. **Computational requirements**: Although more efficient than predecessors, still requires significant computation (4 TPUs, 72 hours) for training.
3. **Single-domain**: Demonstrated only for Go; generalization to other domains requires game-specific adaptations (later addressed by AlphaZero and MuZero).
4. **Discrete action space**: The approach assumes a finite, enumerable action space (all legal Go moves).

## Relevance to CTRA

AlphaGo Zero is a foundational reference for CTRA's MCTS architecture, cited in `README.md` as ref [28]. Key connections:

1. **Neural network-guided MCTS → LLM-guided MCTS**: AlphaGo Zero's central innovation — using a learned policy network to guide MCTS selection instead of uniform random exploration — is the direct archetype for CTRA's approach of using LLM value estimates to guide tree search over feature engineering strategies. Just as AlphaGo Zero's policy head biases search toward promising moves, CTRA's LLM agents bias search toward promising feature transformations.

2. **Informed simulation selection**: The README documents an MCTS fix for "informed simulation selection" — replacing random selection with weighted selection favoring the evaluator's top suggestions. This directly mirrors AlphaGo Zero's use of the policy network prior P(s,a) to bias which branches are explored.

3. **Value network as evaluator**: AlphaGo Zero's value head V(s) that evaluates positions without rollouts parallels CTRA's approach of using cross-validation scores to evaluate feature sets without full pipeline execution.

4. **Self-play improvement loop**: The iterative self-improvement cycle (better network → better self-play → better training data → better network) maps to CTRA's MCTS loop where better feature proposals → better models → better reward signals → better feature proposals.

## Code/Data Availability

- **Code**: Not publicly released by DeepMind
- **Open-source reimplementations**: Leela Zero (https://github.com/leela-zero/leela-zero), KataGo (https://github.com/lightvector/KataGo)
- **Training data**: Self-generated (no external data needed)

## Citation

```bibtex
@article{silver2017mastering,
  title={Mastering the game of Go without human knowledge},
  author={Silver, David and Schrittwieser, Julian and Simonyan, Karen and Antonoglou, Ioannis and Huang, Aja and Guez, Arthur and Hubert, Thomas and Baker, Lucas and Lai, Matthew and Bolton, Adrian and others},
  journal={Nature},
  volume={550},
  number={7676},
  pages={354--359},
  year={2017},
  publisher={Nature Publishing Group},
  doi={10.1038/nature24270}
}
```
