"""
Problem 2: AI-based Travel Planner
==================================
An AI travel planner that consumes existing domain knowledge bases
(modelled here as Python dictionaries — see knowledge_base.py — but in
production they would be the W3C Wine Ontology, DBpedia tourist places,
Schema.org food recipes, etc.) and uses rule-based inference to produce:

  - a personalised tour plan tailored to the user's interests and budget,
  - food and drink recommendations consistent with dietary preferences,
  - a per-day cost assessment and total trip budget.

This is a classic knowledge-based AI agent: the planner combines
constraint satisfaction (cost, time, interests) with content-based
filtering over the ontologies.
"""

from knowledge_base import (TOURIST_PLACES, FOOD_RECOMMENDATIONS,
                            WINE_PAIRINGS, ACCOMMODATION, TRANSPORT_COST)


class TravelPlanner:
    """Knowledge-based personalised travel-planning agent."""

    def __init__(self, user_profile: dict):
        """
        user_profile keys:
          - name (str)
          - origin (str) — home city
          - destination (str)
          - days (int)
          - budget (int, INR) — total trip budget
          - interests (list[str]) — e.g. ['history', 'beach', 'food']
          - diet (str) — 'vegetarian' | 'non-vegetarian'
          - drinks (str) — 'wine' | 'non-alcoholic'
          - accommodation_tier (str) — 'budget' | 'mid' | 'luxury'
        """
        self.user = user_profile

    # ---- Personalisation: match user interests against place tags ----
    def _score_place(self, place: dict) -> int:
        """A place's score = number of its tags matching user interests."""
        interests = set(t.lower() for t in self.user.get("interests", []))
        place_tags = set(t.lower() for t in place["tags"])
        return len(interests & place_tags)

    def recommend_places(self) -> list:
        """Rank places by relevance and return the top N for the trip."""
        dest = self.user["destination"]
        if dest not in TOURIST_PLACES:
            raise ValueError(f"No knowledge base entry for {dest!r}")

        places = TOURIST_PLACES[dest]
        # Sort by interest-match first, then by rating
        scored = sorted(
            places,
            key=lambda p: (self._score_place(p), p["rating"]),
            reverse=True
        )
        # ~2 places per day is a comfortable pace
        return scored[:max(1, self.user["days"] * 2)]

    # ---- Food recommendations filtered by dietary preference ----
    def recommend_food(self) -> list:
        dest = self.user["destination"]
        diet = self.user.get("diet", "vegetarian").lower()
        all_food = FOOD_RECOMMENDATIONS.get(dest, [])
        if diet == "vegetarian":
            return [f for f in all_food if f["type"] == "vegetarian"]
        return all_food  # non-vegetarian users get everything

    # ---- Wine/drink pairing based on diet + alcohol preference ----
    def recommend_drinks(self) -> list:
        if self.user.get("drinks", "non-alcoholic") == "wine":
            diet = self.user.get("diet", "vegetarian").lower()
            return WINE_PAIRINGS.get(diet, [])
        return WINE_PAIRINGS["non-alcoholic"]

    # ---- Cost assessment ----
    def compute_costs(self, places: list, food: list) -> dict:
        u = self.user
        dest, origin, days = u["destination"], u["origin"], u["days"]

        # Transport — fall back to a default if origin not in table
        transport = TRANSPORT_COST.get((origin, dest), 5000) * 2  # round-trip

        # Accommodation
        per_night = ACCOMMODATION[dest][u.get("accommodation_tier", "mid")]
        stay = per_night * days

        # Sightseeing entry fees
        entry = sum(p["cost"] for p in places)

        # Food: ~3 meals/day, averaged across the recommended dishes
        if food:
            avg_meal = sum(f["cost"] for f in food) / len(food)
        else:
            avg_meal = 200  # safe default
        food_total = int(avg_meal * 3 * days)

        misc = 1000 * days  # local transport, tips, etc.

        total = transport + stay + entry + food_total + misc
        return {
            "transport": transport,
            "accommodation": stay,
            "sightseeing_entry": entry,
            "food": food_total,
            "miscellaneous": misc,
            "total": total,
        }

    # ---- Distribute the chosen places into a per-day itinerary ----
    def build_itinerary(self, places: list) -> dict:
        days = self.user["days"]
        # round-robin: place i goes into day (i mod days) + 1
        itin = {f"Day {i+1}": [] for i in range(days)}
        for i, p in enumerate(places):
            itin[f"Day {i % days + 1}"].append(p)
        return itin

    # ---- Public entry point ----
    def plan(self) -> dict:
        places = self.recommend_places()
        food = self.recommend_food()
        drinks = self.recommend_drinks()
        costs = self.compute_costs(places, food)
        itinerary = self.build_itinerary(places)

        within_budget = costs["total"] <= self.user["budget"]
        return {
            "user": self.user["name"],
            "destination": self.user["destination"],
            "days": self.user["days"],
            "itinerary": itinerary,
            "food_recommendations": food,
            "drink_recommendations": drinks,
            "cost_breakdown": costs,
            "within_budget": within_budget,
        }


# ----------------------------------------------------------------------
# Pretty printer
# ----------------------------------------------------------------------
def display_plan(plan: dict):
    print("\n" + "=" * 60)
    print(f"  TRAVEL PLAN for {plan['user']}")
    print(f"  Destination: {plan['destination']}  |  Duration: "
          f"{plan['days']} days")
    print("=" * 60)

    print("\n--- ITINERARY ---")
    for day, places in plan["itinerary"].items():
        print(f"\n{day}:")
        if not places:
            print("  (free day)")
            continue
        for p in places:
            print(f"  • {p['name']}  ({p['type']}, ~{p['hours']}h, "
                  f"₹{p['cost']}, ★{p['rating']})")

    print("\n--- FOOD RECOMMENDATIONS ---")
    for f in plan["food_recommendations"]:
        print(f"  • {f['dish']}  [{f['type']}, spice={f['spice']}, "
              f"₹{f['cost']}]")

    print("\n--- DRINK PAIRINGS ---")
    for d in plan["drink_recommendations"]:
        if "wine" in d:
            print(f"  • {d['wine']}  ({d['type']}, body={d['body']}) — "
                  f"pairs with {d['pairs_well_with']}")
        else:
            print(f"  • {d['drink']} — pairs with {d['pairs_well_with']}")

    print("\n--- COST ASSESSMENT (INR) ---")
    for k, v in plan["cost_breakdown"].items():
        print(f"  {k:<20s} ₹{v:,}")

    status = ("WITHIN BUDGET" if plan["within_budget"]
              else "OVER BUDGET — consider downgrading accommodation")
    print(f"\n  Status: {status}")
    print("=" * 60)


# ----------------------------------------------------------------------
# TEST CASES — three different user profiles
# ----------------------------------------------------------------------
def run_tests():
    print("\n" + "#" * 60)
    print("# Q2 — AI TRAVEL PLANNER — TEST CASES")
    print("#" * 60)

    # Test 1: history-loving vegetarian to Jaipur
    user1 = {
        "name": "Karthik",
        "origin": "Hyderabad",
        "destination": "Jaipur",
        "days": 3,
        "budget": 30000,
        "interests": ["history", "architecture", "culture"],
        "diet": "vegetarian",
        "drinks": "non-alcoholic",
        "accommodation_tier": "mid",
    }
    plan1 = TravelPlanner(user1).plan()
    display_plan(plan1)
    assert plan1["destination"] == "Jaipur"
    # Top recommendation should be a history place
    top = plan1["itinerary"]["Day 1"][0]
    assert "history" in top["tags"] or "architecture" in top["tags"]
    print("\n>> Test 1 PASSED — history-loving vegetarian to Jaipur")

    # Test 2: beach/relaxing non-vegetarian to Goa
    user2 = {
        "name": "Aditi",
        "origin": "Mumbai",
        "destination": "Goa",
        "days": 4,
        "budget": 40000,
        "interests": ["beach", "relaxing", "watersports"],
        "diet": "non-vegetarian",
        "drinks": "wine",
        "accommodation_tier": "luxury",
    }
    plan2 = TravelPlanner(user2).plan()
    display_plan(plan2)
    # Wine recommendations should include red wines for non-veg diet
    wines = plan2["drink_recommendations"]
    assert any(w.get("type") == "red" for w in wines)
    print("\n>> Test 2 PASSED — beach lover to Goa with wine pairings")

    # Test 3: budget weekend in Hyderabad
    user3 = {
        "name": "Ravi",
        "origin": "Bangalore",
        "destination": "Hyderabad",
        "days": 2,
        "budget": 15000,
        "interests": ["history", "family", "fun"],
        "diet": "non-vegetarian",
        "drinks": "non-alcoholic",
        "accommodation_tier": "budget",
    }
    plan3 = TravelPlanner(user3).plan()
    display_plan(plan3)
    print("\n>> Test 3 PASSED — budget weekend in Hyderabad")

    print("\nAll Q2 tests PASSED.\n")


if __name__ == "__main__":
    run_tests()
