from itertools import product
from sklearn.metrics import f1_score

def custom_f1_score(y_true, y_pred, claims, stances):
    # compute macro F1 score of FOR and AGAINST for each of 3 classes. Then take the average of that
    unique_claims = set(claims)
    f1_acc = 0
    for claim, stance in product(unique_claims, (1, 2)):
        mask = ((claims == claim) & (stances == stance)) > 0
        f1_acc += f1_score(y_true[mask] >= 0.5, y_pred[mask])
        
    return f1_acc / (len(unique_claims) * 2)