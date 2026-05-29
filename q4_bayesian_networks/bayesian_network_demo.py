"""
Problem 4: Bayesian Networks — Modelling, Representation, Inference
====================================================================

A Bayesian Network is a Directed Acyclic Graph (DAG) where:
  - Each node is a random variable.
  - Each directed edge (X -> Y) means "X is a direct cause of Y".
  - Each node carries a Conditional Probability Table (CPT) that
    quantifies P(node | parents(node)).

The joint distribution factorises as:
    P(X1, ..., Xn) = ∏_i  P(Xi | parents(Xi))

This compact representation lets us reason about uncertain domains.
Given evidence on some variables, we can use *inference* to compute the
posterior probability of any query variable. Two common algorithms:
  - Variable Elimination — exact inference, polynomial in tree-width.
  - MCMC sampling        — approximate inference for large networks.

TOOLS COMMONLY USED FOR BAYESIAN NETWORKS
-----------------------------------------
  1. pgmpy        — pure-Python; supports modelling, exact & approximate
                    inference, parameter & structure learning. (Used here.)
  2. PyMC         — probabilistic programming for continuous + discrete
                    models, MCMC and variational inference.
  3. Pyro / Stan  — production-grade probabilistic programming.
  4. BayesFusion GeNIe / SMILE — GUI + C++ engine, widely used in industry.
  5. Hugin        — commercial Bayesian Network toolkit.
  6. Netica       — another commercial BN tool.
  7. bnlearn (R)  — gold-standard R package for structure learning.

EXAMPLE IMPLEMENTED HERE
------------------------
Russell & Norvig's classic Burglary–Alarm network (AIMA, Ch. 14):

        Burglary       Earthquake
             \\           /
              \\         /
               v       v
                Alarm
               /     \\
              v       v
        JohnCalls   MaryCalls

  - The alarm can be triggered by either a burglary or a small earthquake.
  - John and Mary both phone you when (and only when) they hear the alarm,
    but they are not perfectly reliable.

We use it to demonstrate:
  - building the network and CPTs in pgmpy,
  - exact inference with Variable Elimination,
  - diagnostic ("Given that John called, was there a burglary?") and
    causal ("Given a burglary, will Mary call?") reasoning,
  - explaining away (multiple causes competing to explain evidence).
"""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# ----------------------------------------------------------------------
# 1. STRUCTURE — the DAG of causal relationships
# ----------------------------------------------------------------------
def build_alarm_network() -> DiscreteBayesianNetwork:
    model = DiscreteBayesianNetwork([
        ("Burglary",   "Alarm"),
        ("Earthquake", "Alarm"),
        ("Alarm",      "JohnCalls"),
        ("Alarm",      "MaryCalls"),
    ])

    # ------------------------------------------------------------------
    # 2. PARAMETERS — Conditional Probability Tables (Russell & Norvig)
    # ------------------------------------------------------------------
    # Prior: burglary is rare (1 in 1000 nights)
    cpd_burglary = TabularCPD(
        variable="Burglary",
        variable_card=2,
        values=[[0.999],   # P(B=False)
                [0.001]],  # P(B=True)
        state_names={"Burglary": ["False", "True"]},
    )

    # Prior: small earthquake is even rarer for the alarm to react to
    cpd_earthquake = TabularCPD(
        variable="Earthquake",
        variable_card=2,
        values=[[0.998],   # P(E=False)
                [0.002]],  # P(E=True)
        state_names={"Earthquake": ["False", "True"]},
    )

    # Alarm conditional on its two parents.
    #
    # Column ordering follows pgmpy convention: parents iterate fastest on
    # the right. With evidence=[Burglary, Earthquake]:
    #   col 0: B=False, E=False
    #   col 1: B=False, E=True
    #   col 2: B=True,  E=False
    #   col 3: B=True,  E=True
    cpd_alarm = TabularCPD(
        variable="Alarm",
        variable_card=2,
        values=[
            # P(A=False | B,E)
            [0.999, 0.71, 0.06, 0.05],
            # P(A=True  | B,E)
            [0.001, 0.29, 0.94, 0.95],
        ],
        evidence=["Burglary", "Earthquake"],
        evidence_card=[2, 2],
        state_names={
            "Alarm": ["False", "True"],
            "Burglary": ["False", "True"],
            "Earthquake": ["False", "True"],
        },
    )

    cpd_john = TabularCPD(
        variable="JohnCalls",
        variable_card=2,
        values=[
            [0.95, 0.10],   # P(J=False | A)
            [0.05, 0.90],   # P(J=True  | A)
        ],
        evidence=["Alarm"],
        evidence_card=[2],
        state_names={
            "JohnCalls": ["False", "True"],
            "Alarm": ["False", "True"],
        },
    )

    cpd_mary = TabularCPD(
        variable="MaryCalls",
        variable_card=2,
        values=[
            [0.99, 0.30],   # P(M=False | A)
            [0.01, 0.70],   # P(M=True  | A)
        ],
        evidence=["Alarm"],
        evidence_card=[2],
        state_names={
            "MaryCalls": ["False", "True"],
            "Alarm": ["False", "True"],
        },
    )

    model.add_cpds(cpd_burglary, cpd_earthquake, cpd_alarm,
                   cpd_john, cpd_mary)
    assert model.check_model(), "Model failed validation!"
    return model


# ----------------------------------------------------------------------
# 3. INFERENCE QUERIES
# ----------------------------------------------------------------------
def run_tests():
    print("\n" + "#" * 60)
    print("# Q4 — BAYESIAN NETWORKS (Burglary-Alarm) — TEST CASES")
    print("#" * 60)

    model = build_alarm_network()
    infer = VariableElimination(model)

    print(f"\nNetwork built. Nodes : {list(model.nodes())}")
    print(f"             Edges : {list(model.edges())}")

    # -- Test 1: marginal probability of a burglary (no evidence) --
    print("\n--- Test 1: prior P(Burglary) ---")
    q = infer.query(variables=["Burglary"], show_progress=False)
    print(q)
    p_true = q.values[q.state_names["Burglary"].index("True")]
    assert abs(p_true - 0.001) < 1e-6
    print("  PASSED — matches the prior we set")

    # -- Test 2: DIAGNOSTIC reasoning --
    print("\n--- Test 2: P(Burglary | JohnCalls=True, MaryCalls=True) ---")
    q = infer.query(
        variables=["Burglary"],
        evidence={"JohnCalls": "True", "MaryCalls": "True"},
        show_progress=False,
    )
    print(q)
    p_burglary = q.values[q.state_names["Burglary"].index("True")]
    print(f"  P(Burglary=True | J,M) = {p_burglary:.4f}")
    # Russell & Norvig report ~0.284 for this query
    assert 0.20 < p_burglary < 0.35, \
        "Expected ~0.28 from AIMA's standard analysis"
    print("  PASSED — both callers raise belief in a burglary "
          f"to ~{p_burglary*100:.1f}% (from 0.1%).")

    # -- Test 3: CAUSAL reasoning --
    print("\n--- Test 3: P(MaryCalls | Burglary=True) ---")
    q = infer.query(
        variables=["MaryCalls"],
        evidence={"Burglary": "True"},
        show_progress=False,
    )
    print(q)
    p_mary = q.values[q.state_names["MaryCalls"].index("True")]
    print(f"  P(MaryCalls=True | Burglary) = {p_mary:.4f}")
    assert p_mary > 0.5, "Burglary should make Mary likely to call"
    print("  PASSED — burglary makes Mary's call probable.")

    # -- Test 4: EXPLAINING AWAY --
    print("\n--- Test 4: explaining-away effect ---")
    q_no = infer.query(
        variables=["Burglary"],
        evidence={"Alarm": "True"},
        show_progress=False,
    )
    p_b_alarm = q_no.values[q_no.state_names["Burglary"].index("True")]

    q_yes = infer.query(
        variables=["Burglary"],
        evidence={"Alarm": "True", "Earthquake": "True"},
        show_progress=False,
    )
    p_b_alarm_eq = q_yes.values[q_yes.state_names["Burglary"].index("True")]

    print(f"  P(Burglary | Alarm)             = {p_b_alarm:.4f}")
    print(f"  P(Burglary | Alarm, Earthquake) = {p_b_alarm_eq:.4f}")
    assert p_b_alarm_eq < p_b_alarm, \
        "Earthquake should explain away the alarm and lower P(Burglary)"
    print("  PASSED — observing the earthquake lowers our belief in "
          "a burglary (the earthquake explains away the alarm).")

    # -- Test 5: joint probability factorisation --
    print("\n--- Test 5: joint probability of a specific full assignment ---")
    q = infer.query(
        variables=["Burglary", "Earthquake", "Alarm",
                   "JohnCalls", "MaryCalls"],
        show_progress=False,
    )
    # Probability of the most common assignment: nothing happens
    idx = (q.state_names["Burglary"].index("False"),
           q.state_names["Earthquake"].index("False"),
           q.state_names["Alarm"].index("False"),
           q.state_names["JohnCalls"].index("False"),
           q.state_names["MaryCalls"].index("False"))
    p_nothing = q.values[idx]
    print(f"  P(everyone-safe-no-alarm) = {p_nothing:.6f}")
    assert p_nothing > 0.9, "Quiet nights should be by far the most common"
    print("  PASSED — quiet nights dominate, as expected.")

    print("\nAll Q4 tests PASSED.\n")


if __name__ == "__main__":
    run_tests()
