from numpy import sum, mean, var, array, sqrt, zeros, diag, std
from numpy.linalg import norm
def vip(x, y, model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_

    m, p = x.shape
    _, h = t.shape

    vips = zeros((p,))

    s = diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = sum(s)

    for i in range(p):
        weight = array([ (w[i,j] / norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = sqrt(p*(s.T @ weight)/total_s)

    return vips

def ccc(x,y):
    
    ''' Concordance Correlation Coefficient'''
    sxy = sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (var(x) + var(y) + (x.mean() - y.mean())**2)
    return rhoc

def r(x,y):
    ''' Pearson Correlation Coefficient'''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rho = sxy / (std(x)*std(y))
    return rho
