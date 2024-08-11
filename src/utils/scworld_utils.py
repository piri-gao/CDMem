import numpy as np
import re

action_type_description = [
    {"action_type": "WAIT()",
     "desc": "wait for something to be done, for example, an object on stove to be boiled"},
    {"action_type": "TELEPORT(room)", "desc": "directly go to a room such as TELEPORT(kitchen)"},
    # {"action_type": "LOOK(object)", "desc": "look at an object"},
    {"action_type": "READ(object)", "desc": "read an object such as a recipe or a book"},
    {"action_type": "PICK(object)", "desc": "pick up an object and put it into your inventory"},
    {"action_type": "OPEN(object)",
     "desc": "open an object with doors before you search or put things in it. For example, OPEN(freezer), OPEN(blast furnace)."},
    {"action_type": "ACTIVATE(object)",
     "desc": "activate and turn on an object such as sink or stove, so that you can use it. "},
    {"action_type": "DEACTIVATE(object)", "desc": "deactivate turn off the object"},
    {"action_type": "EXAMINE(object)",
     "desc": "look at an object carefully. For example, EXAMINE(apple). Note that you cannot EXAMINE a location."},
    {"action_type": "CONNECT(object)", "desc": "connect two objects so that they become useful"},
    {"action_type": "MOVE(object, place)", "desc": "move/place the object to a place"},
    {"action_type": "USE(object A, object B)",
     "desc": "use an object A on object B, for example, USE(thermometer in inventory, water) to check the temperature of water."},
    {"action_type": "MIX(container)",
     "desc": "mix the objects in a container such as MIX(cup containing sugar and water)"},
    {"action_type": "DUNK(object A, object B)", "desc": "dunk object A into object B (optional)"},
    {"action_type": "DROP(object A, object B)", "desc": "drop object A into object B (optional)"},
    {"action_type": "POUR(object A, object B)",
     "desc": "pour the object A into the container B; For example, POUR(red paint, glass cup)"},
    {"action_type": "FOCUS(object)",
     "desc": "focus on an important object that are required by the task description (e.g., a substance, a plant, an animal, and so on)."},
]

focus_on_count = {
    "0": 1, "1": 1, "2": 1, "3": 1, "4": 2, "5": 1, "6": 1, "7": 1,
    "8": 1, "9": 1, "10": 1, "11": 1, "12": 4, "13": 4, "14": 1, "15": 1,
    "16": 1, "17": 1, "18": 2, "19": 1, "20": 3, "21": 3, "22": 1, "23": 1,
    "24": 1, "25": 1, "26": 2, "27": 1, "28": 1, "29": 2

}


def findValidActionNew(predictions, env, look, recent_actions):
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
    for pred in predictions:
        pred = pred.replace("green house", "greenhouse")
        if pred.strip() in validActions:
            found_valid_in_top = True
            action = pred.strip()
            break
    if found_valid_in_top:
        return action
    else:
        validActions = list(validActions)
        validActions.sort(key=lambda x: len(x))

    # jaccard
    topValue = 0.0
    topAction = predictions[0]
    # embPred = sbert_model.encode(pred, convert_to_tensor=True)
    tokensPred = predictions[0].split(" ")
    uniqueTokensPred = set(tokensPred)

    for validAction in validActions:
        tokensAction = validAction.split(" ")
        uniqueTokensAction = set(tokensAction)

        intersection = uniqueTokensPred.intersection(uniqueTokensAction)
        if (len(intersection) > topValue):
            topAction = validAction
            topValue = len(intersection)

    # Sanitize top action
    topAction = re.sub(r'[^A-Za-z0-9 ]+', '', topAction)
    action = topAction
    return action
