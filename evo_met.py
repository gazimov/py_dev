import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams['figure.figsize'] = 12, 12
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

# функция считывания бита по индексу
def getbit(n, i):
    return n & (1 << i) > 0
# функция задания бита по индексу
def setbit(n, i, st):
    if (st): return n | (1<<i)
    else: return n & ~(1<<i)
# функция получения массива битов (генетического кода) координаты
def gencode(nmb):
    n = np.uint32(100000000*nmb)
    x = np.zeros((32, 1), dtype=bool)
    for i in range(32):
        x[i] = getbit(n, i)
    return x
# функция получения координаты из ее генетического кода 
def fencode(gc):
    n = 0
    for i in range(32):
        n = setbit(n, i, gc[i])
    return n/100000000
# функция скрещивания двух координат и мутации их потомка
def crossmut(gc1, gc2, mf):
    cp = np.random.uniform(0,1,32)
    gc = np.zeros((32, 1), dtype=bool)
    for i in range(32):
        if cp[i] > 0.5: gc[i] = gc1[i]
        else: gc[i] = gc2[i]
        if cp[i] < mf: gc[i] = ~gc[i]
    return gc

def minimize(func, rv, bounds, args=None, eps=0.0000001):
    """ Реализует метод поиска минимума генетическим алгоритмом.
    
    Keyword args:
    func --- целевая функция, задается функцией или callable объектом
    x0 --- начальное значение кординат, список размера dims
    params --- параматры поиска - начальные значения шаговых коэффициентов k
    args --- дополнительные параметры целевой функции, словарь 
    {'имя_парамтера':значение}
    eps --- точность поиска
    
    Returns:
    Кортеж из трех элементов: минимум, найденное минимизирующее значение аргументов в 
    виде списка и количество итераций
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
             rv = rv,
             bounds = bounds, 
             args =args,  
             px = pp[:,0],
             py = pp[:,1],
             cnt = cnt,
             met = 1) 
 
    pp = pp[np.argsort(pp[:,2])]
    for i in range(M):
        pb[i,:] = pp[i,:]
    
    f0 = pp[0,2]
    ff = f0+2*eps
    
    while abs(f0 - ff) > eps:
        cnt += 1
        f0 = ff
        
        for i in range(M):
            px1 = gencode(pb[i,0] - bounds[0][0])
            px2 = gencode(pb[i,1] - bounds[1][0])
            
            for j in range(L):
                fl = 0
                while fl == 0:
                    ind = np.random.randint(M-1)
                    mx1 = gencode(pb[ind,0] - bounds[0][0])
                    mx2 = gencode(pb[ind,1] - bounds[1][0])
                    cx1 = crossmut(px1,mx1,0.05)
                    cx2 = crossmut(px2,mx2,0.05)
                    k = L*i+j
                    pp[k,0] = bounds[0][0] + fencode(cx1)
                    pp[k,1] = bounds[1][0] + fencode(cx2)
                    if (pp[k,0] > bounds[0][0]) and (pp[k,0] < bounds[0][1]) and (pp[k,1] > bounds[1][0]) and (pp[k,1] < bounds[1][1]):
                        fl = 1
                pp[k,2] = func(pp[k,0], pp[k,1], **args)
        
        pp = pp[np.argsort(pp[:,2])]
        for i in range(M):
            pb[i,:] = pp[i,:]
        ff = pp[0,2]
        dx = dx/5
        
        func_plt(func = func,
                 rv = rv,
                 bounds = bounds, 
                 args = args, 
                 px = pp[:,0],
                 py = pp[:,1],
                 cnt = cnt,
                 met = 1) 
    
    x[0] = pp[0,0]
    x[1] = pp[0,1] 
    
    return x, ff, cnt

def func_plt(func, rv, bounds, args, px, py, cnt, met):
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

    if met == 1:
        plt.scatter(px, py, s=10, c='r', alpha=0.5)
    elif met == 0:
        plt.plot(px, py,'b-',linewidth=2)
        
    plt.grid(True)
    plt.title("Evolution search (step # {val})".format(val=cnt))
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
