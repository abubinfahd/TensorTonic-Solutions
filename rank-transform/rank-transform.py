def rank_transform(values):
    """
    Replace each value with its average rank.
    """
    n = len(values)
    
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1])
    
    ranks = [0.0] * n
    i = 0
    
    while i < n:
        j = i
        
        while j + 1 < n and indexed[j][1] == indexed[j + 1][1]:
            j += 1
        
        avg_rank = (i + 1 + j + 1) / 2.0
        
        for k in range(i, j + 1):
            original_index = indexed[k][0]
            ranks[original_index] = avg_rank
        
        i = j + 1
    
    return ranks