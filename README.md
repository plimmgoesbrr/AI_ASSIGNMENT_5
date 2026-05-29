
This submission contains complete, self-contained implementations of all
four problems in the programming assignment. Every program prints its
problem statement, runs its test cases, and prints PASS / FAIL diagnostics.

---

## Folder layout

```
ai_assignment/
├── README.md                       <-- this file
├── run_all.sh                      <-- runs every program in one shot
├── q1_search_algorithms/
│   ├── tic_tac_toe.py              shared game-state class
│   ├── minimax.py                  Q1a: Minimax
│   ├── alpha_beta.py               Q1b: Alpha-Beta pruning
│   ├── heuristic_alpha_beta.py     Q1c: Heuristic Alpha-Beta
│   ├── mcts.py                     Q1d: Monte-Carlo Tree Search
│   └── run_all_tests.py            runs all four + a perf comparison
├── q2_travel_planner/
│   ├── knowledge_base.py           tourist / food / wine / cost data
│   └── travel_planner.py           AI travel planner + 3 user profiles
├── q3_knowledge_graphs/
│   ├── knowledge_graph_demo.py     KG build + inference + viz
│   └── knowledge_graph.png         (generated on run)
└── q4_bayesian_networks/
    └── bayesian_network_demo.py    Burglary-Alarm BN with pgmpy
```

---

## Dependencies

```bash
pip install networkx matplotlib pgmpy
```

Python ≥ 3.10 is required. No internet access is needed once the
packages above are installed — every knowledge base used in the
assignment is embedded in code.

---

## Problem 1 — Search Algorithms

Implement Minimax, Alpha-Beta, Heuristic Alpha-Beta, and Monte-Carlo
Tree Search.

All four algorithms operate on the same Tic-Tac-Toe state class
(`tic_tac_toe.py`) so their behaviour can be compared directly.

### Files
| File | What it contains |
|------|-----|
| `minimax.py` | Pure minimax with recursive utility back-up. |
| `alpha_beta.py` | Same logic but with α-β cut-offs. |
| `heuristic_alpha_beta.py` | Depth-limited search with an evaluation function: `(# lines open to X) − (# lines open to O)`. |
| `mcts.py` | UCB1 selection → expansion → random rollout → back-prop, returns the most-visited child. |
| `run_all_tests.py` | Runs every test in the four files plus a performance comparison. |

### Test cases (each algorithm)
1. **Immediate win** — X to move with two-in-a-row; the algorithm must complete the line.
2. **Forced block** — O has a two-in-a-row threat; X must block.
3. **Empty board** — optimal play between two perfect agents results in a draw.

### Performance comparison (empty board, X to move)
| Algorithm | Nodes explored |
|---|---|
| Minimax (full tree) | 549,945 |
| Alpha-Beta | 18,296 |
| Heuristic Alpha-Beta (depth 3) | 162 |
| MCTS (1000 iterations) | n/a (uses rollouts, not full expansion) |

Alpha-Beta prunes ~96.7% of the nodes minimax visits, and the heuristic
cut-off variant explores three orders of magnitude fewer nodes still.

---

## Problem 2 — AI Travel Planner

A knowledge-based travel-planning agent that reuses existing domain
knowledge bases. To keep the assignment self-contained, the planner
uses Python dictionaries that *mirror* real ontologies:

- **Wine pairings** ← W3C Wine Ontology
- **Tourist places** ← DBpedia `TouristAttraction`
- **Food recommendations** ← Schema.org `Recipe` / `FoodEstablishment`

The planner takes a user profile (origin, destination, days, budget,
interests, diet, drink preference, accommodation tier) and produces:

1. A per-day itinerary ranked by interest match and rating.
2. Food recommendations filtered by dietary preference.
3. Drink pairings (wine for non-veg/non-alcoholic, etc.).
4. A complete cost breakdown with a budget check.

### Test cases
1. **Karthik** — vegetarian heritage-lover, Hyderabad → Jaipur, 3 days.
2. **Aditi** — non-vegetarian wine drinker, Mumbai → Goa luxury beach trip.
3. **Ravi** — non-vegetarian budget traveller, Bangalore → Hyderabad.

---

## Problem 3 — Knowledge Graphs

Two parts: a written description plus a working demo.

The description (top of `knowledge_graph_demo.py`) covers what a
knowledge graph is, the triple representation, common applications,
and a list of the major tools used to build KGs: **Neo4j, RDFLib,
Apache Jena, GraphDB, NetworkX, Protégé, owlready2, PyKEEN,
AmpliGraph**.

The demo builds a small KG of Indian tourist destinations as a
`networkx.MultiDiGraph`, then:

1. Performs forward and reverse triple queries.
2. Applies Horn-clause inference rules iteratively to a fixed point:
   - `locatedIn(X,Y) ∧ isCapitalOf(Y,Z) ⇒ locatedIn(X,Z)`
   - `locatedIn(X,Y) ∧ isStateOf(Y,Z) ⇒ locatedIn(X,Z)`
   - `isA(X,Y) ∧ isA(Y,Z) ⇒ isA(X,Z)`
3. Serialises the graph to RDF Turtle.
4. Renders the graph as a PNG (`knowledge_graph.png`) via matplotlib.

After inference, the KG correctly derives that *Charminar* is located
in *Telangana* AND *India*, and that *Amber Fort* is a *Monument*
(via `isA Fort isA Monument`).

---

## Problem 4 — Bayesian Networks

The description in the docstring of `bayesian_network_demo.py` covers
what a Bayesian Network is, how the joint distribution factorises, and
lists popular tools: **pgmpy, PyMC, Pyro, Stan, GeNIe/SMILE, Hugin,
Netica, bnlearn**.

We then implement Russell & Norvig's classic **Burglary-Alarm** network
in pgmpy:

```
   Burglary    Earthquake
        \      /
         v    v
         Alarm
        /     \
       v       v
   JohnCalls  MaryCalls
```

with the standard AIMA CPTs. Five inference queries are run via
**Variable Elimination**:

1. **Prior** — P(Burglary) ≈ 0.001.
2. **Diagnostic** — P(Burglary | John calls, Mary calls) ≈ **0.2842**, matching the AIMA textbook value.
3. **Causal** — P(Mary calls | Burglary) ≈ 0.66.
4. **Explaining away** — observing an earthquake drops P(Burglary | Alarm) from 0.37 down to 0.003.
5. **Joint** — the probability that nothing happens on a given night is ≈ 0.937.

---

## Running everything

From the assignment root:

```bash
bash run_all.sh
```

Or run each problem individually:

```bash
# Problem 1
cd q1_search_algorithms && python3 run_all_tests.py && cd ..

# Problem 2
cd q2_travel_planner && python3 travel_planner.py && cd ..

# Problem 3
cd q3_knowledge_graphs && python3 knowledge_graph_demo.py && cd ..

# Problem 4
cd q4_bayesian_networks && python3 bayesian_network_demo.py && cd ..
```

Every script ends with a clean `All ... tests PASSED.` line if
successful.

---
