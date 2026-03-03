def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_k = recommended[:k]
    count = 0
    for item_i in top_k:
        if item_i in relevant: count+=1
    return [count / k, count / len(relevant)]