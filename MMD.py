import numpy as np

def backDistance(DomainVar):
    X = DomainVar.T
    _,n = X.shape
    G = np.dot(X.T, X)
    H = np.tile(np.diag(G), (n,1))
    M = H + H.T - 2*G
    return M

class Trainer(object):
    def __init__(self, Domain, Variable) -> None:
        super().__init__()
        Domain = Domain.reshape(-1)

        self.Domain = Domain
        self.Variable = Variable

        self.train(Domain, Variable)

    def train(self, Domain=None, Variable=None, mode='mean'):
        if Domain is None or Variable is None: Domain, Variable = self.Domain, self.Variable
        minD = int(np.min(Domain))
        maxD = int(np.max(Domain))
        Dlist = list(range(minD, maxD+1))
        DomainVar = []

        for item in Dlist:
            DomainVar.append(Variable[Domain == item].mean(0))

        DomainVar = np.array(DomainVar)

        M = backDistance(DomainVar)
        MMD = M[np.triu_indices(M.shape[0],k=1)]
        
        if mode == 'max':
            D = MMD.max()
        else:
            D = MMD.mean()

        self.M = M
        self.MMD = MMD
        self.D = D

        self.getInd()

        return D

    def getInd(self, x=None, y=None):
        if x is None or y is None:
            D = self.D
            print("MMD: {:.4f}.".format(D))
            return D
        else:
            D = self.train(x, y)
            print("MMD: {:.4f}.".format(D))
            return D