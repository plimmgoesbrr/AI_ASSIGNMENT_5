"""
Knowledge base for the AI Travel Planner.

This module simulates the role that real ontologies (e.g. the W3C Wine
Ontology, DBpedia "TouristAttraction" class, Schema.org/Food) play in a
production system. In real code these would be loaded from Turtle/OWL
files via owlready2 or rdflib; here we embed them as Python dictionaries
so the assignment is self-contained and runnable without a network.
"""

# ----------------------------------------------------------------------
# Tourist places ontology — modelled on DBpedia's TouristAttraction class
# ----------------------------------------------------------------------
TOURIST_PLACES = {
    "Hyderabad": [
        {"name": "Charminar", "type": "heritage",
         "cost": 50, "hours": 2, "rating": 4.5,
         "tags": ["history", "architecture", "culture"]},
        {"name": "Ramoji Film City", "type": "entertainment",
         "cost": 1500, "hours": 8, "rating": 4.6,
         "tags": ["family", "movies", "fun"]},
        {"name": "Golconda Fort", "type": "heritage",
         "cost": 100, "hours": 3, "rating": 4.4,
         "tags": ["history", "architecture", "hiking"]},
        {"name": "Hussain Sagar Lake", "type": "nature",
         "cost": 50, "hours": 2, "rating": 4.2,
         "tags": ["scenic", "relaxing", "boating"]},
        {"name": "Salar Jung Museum", "type": "museum",
         "cost": 50, "hours": 3, "rating": 4.5,
         "tags": ["art", "history", "indoor"]},
    ],
    "Goa": [
        {"name": "Baga Beach", "type": "beach",
         "cost": 0, "hours": 4, "rating": 4.3,
         "tags": ["beach", "relaxing", "watersports"]},
        {"name": "Basilica of Bom Jesus", "type": "heritage",
         "cost": 0, "hours": 2, "rating": 4.6,
         "tags": ["history", "religion", "architecture"]},
        {"name": "Dudhsagar Falls", "type": "nature",
         "cost": 400, "hours": 6, "rating": 4.7,
         "tags": ["scenic", "hiking", "adventure"]},
        {"name": "Anjuna Flea Market", "type": "shopping",
         "cost": 0, "hours": 3, "rating": 4.1,
         "tags": ["shopping", "culture", "food"]},
    ],
    "Jaipur": [
        {"name": "Amber Fort", "type": "heritage",
         "cost": 200, "hours": 3, "rating": 4.6,
         "tags": ["history", "architecture", "culture"]},
        {"name": "Hawa Mahal", "type": "heritage",
         "cost": 50, "hours": 1, "rating": 4.4,
         "tags": ["history", "architecture", "photography"]},
        {"name": "City Palace", "type": "heritage",
         "cost": 200, "hours": 3, "rating": 4.5,
         "tags": ["history", "royalty", "museum"]},
        {"name": "Jantar Mantar", "type": "heritage",
         "cost": 50, "hours": 2, "rating": 4.3,
         "tags": ["science", "history", "astronomy"]},
    ],
}

# ----------------------------------------------------------------------
# Food recommendations — modelled on Schema.org's Recipe / FoodEstablishment
# ----------------------------------------------------------------------
FOOD_RECOMMENDATIONS = {
    "Hyderabad": [
        {"dish": "Hyderabadi Biryani",
         "type": "non-vegetarian", "spice": "high", "cost": 350},
        {"dish": "Haleem",
         "type": "non-vegetarian", "spice": "medium", "cost": 250},
        {"dish": "Mirchi ka Salan",
         "type": "vegetarian", "spice": "high", "cost": 150},
        {"dish": "Double ka Meetha",
         "type": "vegetarian", "spice": "none", "cost": 100},
        {"dish": "Irani Chai with Osmania Biscuits",
         "type": "vegetarian", "spice": "none", "cost": 80},
    ],
    "Goa": [
        {"dish": "Goan Fish Curry",
         "type": "non-vegetarian", "spice": "medium", "cost": 400},
        {"dish": "Pork Vindaloo",
         "type": "non-vegetarian", "spice": "high", "cost": 450},
        {"dish": "Bebinca",
         "type": "vegetarian", "spice": "none", "cost": 200},
        {"dish": "Prawn Balchao",
         "type": "non-vegetarian", "spice": "high", "cost": 500},
    ],
    "Jaipur": [
        {"dish": "Dal Baati Churma",
         "type": "vegetarian", "spice": "medium", "cost": 300},
        {"dish": "Laal Maas",
         "type": "non-vegetarian", "spice": "very high", "cost": 500},
        {"dish": "Ghewar",
         "type": "vegetarian", "spice": "none", "cost": 150},
        {"dish": "Pyaaz Kachori",
         "type": "vegetarian", "spice": "medium", "cost": 50},
    ],
}

# ----------------------------------------------------------------------
# Wine / drink pairings — modelled on the W3C Wine Ontology
# (subset; real ontology has ~140 wine classes with grape, region,
# colour, body and sugar properties).
# ----------------------------------------------------------------------
WINE_PAIRINGS = {
    "non-vegetarian": [
        {"wine": "Sula Cabernet Sauvignon", "type": "red",
         "body": "full", "pairs_well_with": "red meat, biryani"},
        {"wine": "Grover Shiraz", "type": "red",
         "body": "medium", "pairs_well_with": "spicy curries"},
    ],
    "vegetarian": [
        {"wine": "Sula Chenin Blanc", "type": "white",
         "body": "light", "pairs_well_with": "lentils, mild curries"},
        {"wine": "Fratelli Sauvignon Blanc", "type": "white",
         "body": "light", "pairs_well_with": "salads, sweet dishes"},
    ],
    "non-alcoholic": [
        {"drink": "Masala Chai", "pairs_well_with": "snacks, desserts"},
        {"drink": "Fresh Lime Soda", "pairs_well_with": "spicy food"},
        {"drink": "Mango Lassi", "pairs_well_with": "biryani, curries"},
    ],
}

# ----------------------------------------------------------------------
# Accommodation costs (INR per night) — keyed by destination and tier
# ----------------------------------------------------------------------
ACCOMMODATION = {
    "Hyderabad": {"budget": 1500, "mid": 4000, "luxury": 12000},
    "Goa":       {"budget": 2000, "mid": 5500, "luxury": 18000},
    "Jaipur":    {"budget": 1800, "mid": 4500, "luxury": 15000},
}

# ----------------------------------------------------------------------
# Approximate transport cost FROM the user's home city. In production
# this would query a flight/train API (Amadeus, IRCTC, etc.).
# ----------------------------------------------------------------------
TRANSPORT_COST = {
    ("Hyderabad", "Hyderabad"): 0,
    ("Hyderabad", "Goa"):       6000,
    ("Hyderabad", "Jaipur"):    7500,
    ("Bangalore", "Hyderabad"): 4500,
    ("Bangalore", "Goa"):       5000,
    ("Bangalore", "Jaipur"):    8000,
    ("Mumbai",    "Hyderabad"): 5000,
    ("Mumbai",    "Goa"):       3500,
    ("Mumbai",    "Jaipur"):    5500,
}
