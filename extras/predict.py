def predict(VS, x):
    """TODO lecture 1 on introduction page 4"""
    if len(VS) == 0:
        raise ValueError("No hypothesis!")
    if all(h(x) for h in VS):
        return "All positive"
    elif not any(h(x) for h in VS): # any so alike OR, any() only need one positive h(x)
        return "All negative" # not any(), not even one positive h(x)
    else:
        positive_count = sum([h(x) for h in VS])
        return "positive count:{}, negative count:{}".format(positive_count, len(VS) - positive_count)