def normalize(code):
    # Extract only ICD code before ':' and strip spaces
    return code.split(":")[0].strip()


def exact_match(pred, gold):
    return int(normalize(pred) == normalize(gold))


def prefix_match(pred, gold, n=3):
    return int(normalize(pred)[:n] == normalize(gold)[:n])


def evaluate(pred_list, gold_list):
    if not gold_list:
        return {"Exact": 0, "Prefix-3": 0}

    exact = sum(exact_match(p, g) for p, g in zip(pred_list, gold_list)) / len(gold_list)
    prefix3 = sum(prefix_match(p, g, 3) for p, g in zip(pred_list, gold_list)) / len(gold_list)

    return {
        "Exact": round(exact, 4),
        "Prefix-3": round(prefix3, 4)
    }
