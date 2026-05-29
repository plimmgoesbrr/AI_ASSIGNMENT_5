#!/usr/bin/env bash
# Master runner — executes every program in the assignment.
# Use:   bash run_all.sh
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "  Q1 — Search Algorithms"
echo "============================================================"
cd "$ROOT/q1_search_algorithms"
python3 run_all_tests.py

echo
echo "============================================================"
echo "  Q2 — AI Travel Planner"
echo "============================================================"
cd "$ROOT/q2_travel_planner"
python3 travel_planner.py

echo
echo "============================================================"
echo "  Q3 — Knowledge Graphs"
echo "============================================================"
cd "$ROOT/q3_knowledge_graphs"
python3 knowledge_graph_demo.py

echo
echo "============================================================"
echo "  Q4 — Bayesian Networks"
echo "============================================================"
cd "$ROOT/q4_bayesian_networks"
python3 bayesian_network_demo.py

echo
echo "============================================================"
echo "  ALL PROBLEMS COMPLETED SUCCESSFULLY"
echo "============================================================"
