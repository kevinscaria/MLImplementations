def accuracy_score(y_true, y_pred):
    cnt=0
    for i, j in zip(y_true, y_pred):
        if i==j:
            cnt+=1
    return cnt/len(y_pred)