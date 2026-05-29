"""
Problem 3: Knowledge Graphs — Description and Hands-On Demo
===========================================================

WHAT IS A KNOWLEDGE GRAPH?
--------------------------
A Knowledge Graph (KG) is a structured representation of real-world
entities and the relationships between them. Knowledge is stored as a
collection of (subject, predicate, object) triples, often called RDF
triples after the W3C standard.

  Example triples:
    (Hyderabad,   isCapitalOf,  Telangana)
    (Telangana,   isStateOf,    India)
    (Charminar,   locatedIn,    Hyderabad)
    (Charminar,   builtBy,      Muhammad_Quli_Qutb_Shah)

Stored together, these triples form a directed labelled multigraph
that can be queried, traversed, and reasoned over.

KEY PROPERTIES
--------------
  * Entities  — nodes in the graph (people, places, concepts).
  * Relations — labelled directed edges between entities.
  * Schema / ontology — declares allowed entity classes, relation
                        types, and constraints (e.g. domain & range).
  * Inference rules — let new facts be deduced from existing ones, e.g.
                      if (X, isCapitalOf, Y) and (Y, isStateOf, Z) then
                      (X, isCityIn, Z).

APPLICATIONS
------------
  - Google Knowledge Graph (the "info boxes" in search results)
  - Wikidata, DBpedia, YAGO (open public KGs)
  - Recommendation systems, fraud detection, biomedical research
  - Question answering & enterprise data integration

POPULAR TOOLS FOR BUILDING KGs
------------------------------
  1. Neo4j         — property-graph database with the Cypher query
                     language. Industry standard for production KGs.
  2. RDFLib        — pure-Python library for RDF triples; supports
                     Turtle, N-Triples, and SPARQL queries.
  3. Apache Jena   — Java framework: storage, SPARQL endpoint, OWL
                     reasoner.
  4. GraphDB       — commercial RDF triple store with built-in
                     OWL inference.
  5. NetworkX      — generic Python graph library; convenient for
                     small/in-memory KGs and visualisation.
  6. Protégé       — open-source GUI editor for OWL ontologies.
  7. owlready2     — Python library for loading and reasoning over OWL
                     ontologies (used in Q2 to back the wine ontology).
  8. PyKEEN / AmpliGraph — knowledge-graph embedding libraries for ML.

THIS DEMO
---------
We build a small KG about Indian tourist destinations using NetworkX,
add inference rules to derive new facts, run a few queries, and emit
a PNG visualisation. We also dump the same KG in RDF Turtle (text) so
the parallel to a "real" semantic-web KG is explicit.
"""

import os
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # headless backend — needed in containers
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. BUILD THE KNOWLEDGE GRAPH AS A DIRECTED MULTIGRAPH
# ----------------------------------------------------------------------
def build_kg() -> nx.MultiDiGraph:
    """Construct a small KG of Indian tourist destinations."""
    g = nx.MultiDiGraph()

    # Triples: (subject, predicate, object)
    triples = [
        # geographic facts
        ("Hyderabad", "isCapitalOf", "Telangana"),
        ("Telangana", "isStateOf",   "India"),
        ("Jaipur",    "isCapitalOf", "Rajasthan"),
        ("Rajasthan", "isStateOf",   "India"),
        ("Panaji",    "isCapitalOf", "Goa"),
        ("Goa",       "isStateOf",   "India"),

        # monuments and their locations
        ("Charminar",       "locatedIn", "Hyderabad"),
        ("Golconda_Fort",   "locatedIn", "Hyderabad"),
        ("Amber_Fort",      "locatedIn", "Jaipur"),
        ("Hawa_Mahal",      "locatedIn", "Jaipur"),
        ("Basilica",        "locatedIn", "Panaji"),

        # type information
        ("Charminar",     "isA", "Monument"),
        ("Golconda_Fort", "isA", "Fort"),
        ("Amber_Fort",    "isA", "Fort"),
        ("Hawa_Mahal",    "isA", "Monument"),
        ("Basilica",      "isA", "Church"),
        ("Fort",          "isA", "Monument"),
        ("Church",        "isA", "Monument"),

        # builders (historical facts)
        ("Charminar",     "builtBy", "Muhammad_Quli_Qutb_Shah"),
        ("Amber_Fort",    "builtBy", "Raja_Man_Singh"),
        ("Hawa_Mahal",    "builtBy", "Sawai_Pratap_Singh"),
        ("Golconda_Fort", "builtBy", "Kakatiya_dynasty"),
    ]

    for s, p, o in triples:
        g.add_edge(s, o, predicate=p)

    return g


# ----------------------------------------------------------------------
# 2. INFERENCE RULES — derive new triples from existing ones
# ----------------------------------------------------------------------
def apply_inference(g: nx.MultiDiGraph) -> int:
    """
    Two simple Horn-clause-style rules:
        R1: locatedIn(X, Y) AND isCapitalOf(Y, Z)  =>  locatedIn(X, Z)
        R2: locatedIn(X, Y) AND isStateOf(Y, Z)    =>  locatedIn(X, Z)
        R3: isA(X, Y) AND isA(Y, Z)                =>  isA(X, Z)  (transitive)
    Returns the number of newly added edges.
    """
    new_edges = []

    def edge_predicates(u, v):
        if not g.has_edge(u, v):
            return []
        return [d["predicate"] for d in g.get_edge_data(u, v).values()]

    nodes = list(g.nodes())
    for x in nodes:
        for y in list(g.successors(x)):
            for pred1 in edge_predicates(x, y):
                if pred1 != "locatedIn":
                    continue
                for z in list(g.successors(y)):
                    for pred2 in edge_predicates(y, z):
                        if pred2 in ("isCapitalOf", "isStateOf"):
                            if "locatedIn" not in edge_predicates(x, z):
                                new_edges.append((x, z, "locatedIn"))

    # transitive isA
    for x in nodes:
        for y in list(g.successors(x)):
            if "isA" not in edge_predicates(x, y):
                continue
            for z in list(g.successors(y)):
                if "isA" in edge_predicates(y, z):
                    if "isA" not in edge_predicates(x, z):
                        new_edges.append((x, z, "isA"))

    for s, o, p in new_edges:
        g.add_edge(s, o, predicate=p)
    return len(new_edges)


# ----------------------------------------------------------------------
# 3. QUERIES (think of these as SPARQL queries against the KG)
# ----------------------------------------------------------------------
def query_objects(g: nx.MultiDiGraph, subject: str, predicate: str) -> list:
    """Return all O such that the triple (subject, predicate, O) exists."""
    if subject not in g:
        return []
    results = []
    for _, o, data in g.out_edges(subject, data=True):
        if data["predicate"] == predicate:
            results.append(o)
    return results


def query_subjects(g: nx.MultiDiGraph, predicate: str, obj: str) -> list:
    """Return all S such that the triple (S, predicate, obj) exists."""
    if obj not in g:
        return []
    results = []
    for s, _, data in g.in_edges(obj, data=True):
        if data["predicate"] == predicate:
            results.append(s)
    return results


# ----------------------------------------------------------------------
# 4. RDF TURTLE SERIALISATION (pretty-prints the KG in standard format)
# ----------------------------------------------------------------------
def to_turtle(g: nx.MultiDiGraph) -> str:
    lines = ["@prefix ex: <http://example.org/> ."]
    for u, v, data in g.edges(data=True):
        lines.append(f"ex:{u} ex:{data['predicate']} ex:{v} .")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 5. VISUALISATION — write a PNG of the graph
# ----------------------------------------------------------------------
def visualize(g: nx.MultiDiGraph, out_path: str = "knowledge_graph.png"):
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(g, seed=42, k=1.5)
    nx.draw_networkx_nodes(g, pos, node_color="lightblue",
                           node_size=2000, alpha=0.9)
    nx.draw_networkx_labels(g, pos, font_size=8)
    nx.draw_networkx_edges(g, pos, edge_color="gray",
                           arrows=True, arrowsize=15)
    edge_labels = {(u, v): data["predicate"]
                   for u, v, data in g.edges(data=True)}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                 font_size=6)
    plt.title("Indian Tourist Destinations — Knowledge Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


# ----------------------------------------------------------------------
# 6. TEST CASES
# ----------------------------------------------------------------------
def run_tests():
    print("\n" + "#" * 60)
    print("# Q3 — KNOWLEDGE GRAPHS — DEMO & TEST CASES")
    print("#" * 60)

    g = build_kg()
    print(f"\nKG built: {g.number_of_nodes()} entities, "
          f"{g.number_of_edges()} relations.")
    assert g.number_of_nodes() > 10
    assert g.number_of_edges() > 15

    print("\n--- Test 1: Direct query — where is Charminar located? ---")
    locs = query_objects(g, "Charminar", "locatedIn")
    print(f"  Charminar locatedIn -> {locs}")
    assert "Hyderabad" in locs
    print("  PASSED")

    print("\n--- Test 2: Reverse query — what is located in Jaipur? ---")
    subs = query_subjects(g, "locatedIn", "Jaipur")
    print(f"  ? locatedIn Jaipur -> {subs}")
    assert "Amber_Fort" in subs and "Hawa_Mahal" in subs
    print("  PASSED")

    print("\n--- Test 3: Inference — apply Horn-clause rules to fixed point ---")
    total_added = 0
    while True:
        added = apply_inference(g)
        total_added += added
        if added == 0:
            break
    print(f"  Inferred {total_added} new triples (fixed-point reached).")
    new_locs = query_objects(g, "Charminar", "locatedIn")
    print(f"  Charminar locatedIn (after inference) -> {new_locs}")
    assert "Telangana" in new_locs, \
        "Should infer Charminar is in Telangana"
    assert "India" in new_locs, \
        "Should infer Charminar is in India (transitive)"
    print("  PASSED — transitive location inference works")

    # transitive isA inference
    types = query_objects(g, "Amber_Fort", "isA")
    print(f"  Amber_Fort isA (after inference) -> {types}")
    assert "Monument" in types, "Amber_Fort isA Fort isA Monument"
    print("  PASSED — transitive type inference works")

    print("\n--- Test 4: Serialise to RDF Turtle ---")
    turtle = to_turtle(g)
    # Print just the first few lines as a sample
    print("  " + "\n  ".join(turtle.splitlines()[:6]))
    print("  ...")
    print(f"  ({len(turtle.splitlines())} total turtle lines)")

    print("\n--- Test 5: Visualisation ---")
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "knowledge_graph.png")
    visualize(g, out_path)
    assert os.path.exists(out_path)
    print(f"  Image written to: {out_path}")
    print("  PASSED")

    print("\nAll Q3 tests PASSED.\n")


if __name__ == "__main__":
    run_tests()
