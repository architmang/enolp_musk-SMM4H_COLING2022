
from sklearn.metrics import f1_score

def custom_f1_score(y_true, y_pred, claims):
    # compute macro F1 score of FOR and AGAINST for each of 3 classes. Then take the average of that
    unique_claims = set(claims)
    f1_acc = 0
    for claim in unique_claims:
        mask = (claims == claim) > 0
        f1_acc += f1_score(y_true[mask], y_pred[mask], labels=[0, 1], average='macro')
        
    return f1_acc / len(unique_claims)