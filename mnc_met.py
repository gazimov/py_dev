import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams['figure.figsize'] = 12, 12
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

def minimize(func, rv, bounds, args=None, eps=0.0000001):
    """ Реализует метод поиска минимума наискорейшим спуском.
    
    Keyword args:
    func --- целевая функция, задается функцией или callable объектом
    x0 --- начальное значение кординат, список размера dims
    params --- параматры поиска - начальные значения шаговых коэффициентов k
    args --- дополнительные параметры целевой функции, словарь 
    {'имя_парамтера':значение}
    eps --- точность поиска
    
    Returns:
    Кортеж из трех элементов: найденное минимизирующее значение аргументов в 
    виде списка, траектория поиска, и количество итераций
    """
    if not args: args = {}    
    cnt = 0
    x = [0,0]
    
    N = 1000
    M = 50
    L = int(N/M)
    dx = 2    
    
    pp = np.zeros((N,3))
    pb = np.zeros((M,3))
    
    pp[:,0] = np.random.uniform(bounds[0][0],bounds[0][1],N)
    pp[:,1] = np.random.uniform(bounds[1][0],bounds[1][1],N)
    pp[:,2] = func(pp[:,0], pp[:,1], **args)
    
    func_plt(func = func, 
             bounds = bounds, 
             args =args, 
             rv = rv, 
             px = pp[:,0],
             py = pp[:,1],
             cnt = cnt) 
 
    pp = pp[np.argsort(pp[:,2])]
    for i in range(M):
        pb[i,:] = pp[i,:]
    
    f0 = pp[0,2]
    ff = f0+2*eps
    
    while abs(f0 - ff) > eps:
        cnt += 1
        f0 = ff
        
        for i in range(M):
            x[0] = pb[i,0]
            x[1] = pb[i,1]
            for j in range(L):
                k = L*i+j
                pp[k,0] = np.random.uniform(x[0]-dx,x[0]+dx)
                pp[k,1] = np.random.uniform(x[1]-dx,x[1]+dx)
                pp[k,2] = func(pp[k,0], pp[k,1], **args)
        
        pp = pp[np.argsort(pp[:,2])]
        for i in range(M):
            pb[i,:] = pp[i,:]
        ff = pp[0,2]
        dx = dx/5
        
        func_plt(func = func, 
                 bounds = bounds, 
                 args =args, 
                 rv = rv, 
                 px = pp[:,0],
                 py = pp[:,1],
                 cnt = cnt) 
    
    x[0] = pp[0,0]
    x[1] = pp[0,1] 
    
    return x, ff, cnt

def func_plt(func, bounds, args, rv, px, py, cnt):
    delta = 0.025
    x = np.arange(bounds[0][0], bounds[0][1], delta)
    y = np.arange(bounds[1][0], bounds[1][1], delta)
    X, Y = np.meshgrid(x, y)
    if not args: args = {}
    Z = func(X,Y, **args)    
    plt.figure()
    lc = -2*abs(rv[2])/(rv[0] - rv[1])
    cs = plt.contour(X, Y, Z, 
                     levels = np.arange(rv[0], rv[1], rv[2]), 
                     linewidths = np.arange(2.1, 0.1, lc), 
                     colors = 'k')
    plt.clabel(cs, inline=1, fontsize=7)

    plt.scatter(px, py, s=10, c='r', alpha=0.5)
    plt.grid(True)
    plt.title("Monte Carlo (step # {val})".format(val=cnt))
    plt.show()

def main():  
    #Пользовательские функции
    def fx1(x, y, verts):
        return -sum([1/(1+(x-v[0])**2+(y-v[1])**2) for v in verts])

    def fx2(x, y, prm):
        return prm[0]*x**2 + prm[1]*y**2 + prm[2]*x*y + prm[3]*y + prm[4]*np.sin(y) 

    def fx3(x, y):
        return x**2 + y**2
    
    class Fx4(object):
        def __init__(self, shift):
            self.shift = shift
        def __call__(self, x, y):
            return (x-self.shift[0])**2 + (y-self.shift[0])**2
    # callable объект 
    fx4 = Fx4(shift=[1,2])
    fncs = [
        {
            'func': fx1, 
            'args': {'verts': [(5,3),(7,5),(15,15)]}, 
            'bnds': [(0.0, 20.0), (0.0, 20.0)], 
            'rv': [0,-1.2,-0.05],
            'init': [10, 1]
        },
        {
            'func': fx2, 
            'args': {'prm': [2,5,4,-3,100]}, 
            'bnds': [(-10.0, 10.0), (-10.0, 10.0)], 
            'rv': [300,-100,-10],
            'init': [1, 1]
        },
        {
            'func': fx3, 
            'args': None, 
            'bnds': [(-10.0, 10.0), (-10.0, 10.0)], 
            'rv': [10, -2, -2],
            'init': [1, 1]
        },
        {
            'func': fx4, 
            'args': None, 
            'bnds': [(-10.0, 10.0), (-10.0, 10.0)], 
            'rv': [10, -2, -2],
            'init': [1, 1]
        }
    ]
    
    if len(sys.argv)<2:
        print("Функция по умолчанию - 0")
        ind = 0
    else:
        ind = int(sys.argv[1])

    if ind > len(fncs)-1:
        print("Превышен индекс, доступны от {0} до {1}".format(0, len(fncs)-1))
        return -1
    
    args_min, min_func, cnt  = minimize(func = fncs[ind]['func'], 
                                              rv = fncs[ind]['rv'], 
                                              bounds = fncs[ind]['bnds'],
                                              args = fncs[ind]['args'])    
    
    print("min = {val}, argmin(f) = {args}, iterations = {i}".format(val=min_func,
                                                                     args=args_min, 
                                                                     i=cnt))  
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
