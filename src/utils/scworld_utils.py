import numpy as np
import re

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

action_type_description = [
    {"action_type": "wait",
     "desc": "wait for something to be done, for example, an object on stove to be boiled. Usage: 'wait#', where # is the number of turns you want to wait. only 'wait' means wait for 10 iterations."},
    {"action_type": "read", "desc": "read an object such as a recipe or a book. Usage: 'read recipe in inventory'"},
    {"action_type": "pick up", "desc": "pick up an object and put it into your inventory. Usage: 'pick up metal pot'"},
    {"action_type": "open",
     "desc": "open an object with doors before you search or put things in it. Usage: 'open door in kitchen', 'open drawer in counter', 'open glass jar'"},
    {"action_type": "activate",
     "desc": "activate and turn on an object such as sink (then the water flow from it) or stove, so that you can use it. Usage: 'activate stove', 'activate sink'"},
    {"action_type": "deactivate", "desc": "deactivate turn off the object"},
    {"action_type": "examine",
     "desc": "look at an object carefully. Note that you cannot examine a location. Usage: 'examine substance in metal pot', 'examine ice'"},
    {"action_type": "move", "desc": "move/place the object to a place. Usage: 'move cupboard to red box'"},
    {"action_type": "use",
     "desc": "use an object A on object B, for example, For example, to check the temperature: Usage: 'use thermometer in inventory on ice', 'use thermometer in inventory on substance in metal pot'"},
    {"action_type": "pour",
     "desc": "pour the object A into the container B. Usage: 'pour jug into flower pot 4'"},
    {"action_type": "focus",
     "desc": "focus on an important object that are required by the task description (e.g., a substance, a plant, an animal, and so on). Usage: 'focus on cupboard'"},
]

focus_on_count = {
    "0": 1, "1": 1, "2": 1, "3": 1, "4": 2, "5": 1, "6": 1, "7": 1,
    "8": 1, "9": 1, "10": 1, "11": 1, "12": 4, "13": 4, "14": 1, "15": 1,
    "16": 1, "17": 1, "18": 2, "19": 1, "20": 3, "21": 3, "22": 1, "23": 1,
    "24": 1, "25": 1, "26": 2, "27": 1, "28": 1, "29": 2

}


def findValidActionNew(predictions, env, look, recent_actions, sbert_model=None):
    rooms = ["hallway", "greenhouse", "green house", "kitchen", "bathroom", "outside", "workshop", "art studio",
             "foundry", "bedroom", "living room"]

    valid_open_door = ["open door to " + i for i in rooms]
    invalid_focus = ["focus on " + x for x in ["agent", "air"] + rooms]
    validActions = set(env.getValidActionObjectCombinations())
    validActions.update(valid_open_door)
    validActions.difference_update(invalid_focus)

    inventory = env.inventory().lower()

    validActions.difference_update(recent_actions[-3:])

    for va in list(validActions):
        if "door" in va and "open" not in va:
            validActions.remove(va)
            continue
        if va.startswith("focus on"):
            pattern = re.compile(r"\b(?:focus|on|in|to)\b", re.IGNORECASE)
            used_objs = pattern.sub("", va).split(" ")
            valid = True
            for obj in used_objs:
                if obj not in look + " " + inventory:
                    valid = False
            if not valid:
                validActions.remove(va)

        # 1) if acton in top k is valid, choose it
        found_valid_in_top = False
        action = None
        for pred in predictions[:5]:
            pred = pred.replace("green house", "greenhouse")
            if pred.strip() in validActions:
                found_valid_in_top = True
                action = pred.strip()
                break
        if found_valid_in_top:
            return action
        else:
            # logger.info(f"No valid action found in top k={k} predictions.")
            validActions = list(validActions)
            validActions.sort(key=lambda x: len(x))
            # logger.info("Valid Predictions: " + str(validActions))

            # 2) else, find most similar action

        return predictions[0]
