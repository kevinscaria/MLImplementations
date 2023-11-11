class NList:
    """
    Helper class to have a restricted circular list
    """
    def __init__(self, n) -> None:
        self.n = n
        self.cnt = 0
        self.data_store = [0]*n

    def append(self, val):
        self.data_store[self.cnt%self.n] = val
        self.cnt+=1

    def is_same(self, ):
         return all(elem == self.data_store[0] for elem in self.data_store)
    
    def __repr__(self):
        return str(self.data_store)