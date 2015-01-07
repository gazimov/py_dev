import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams['figure.figsize'] = 12, 12
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

sign = lambda x: math.copysign(1, x)

def grd(func, x0, args=None, dx=0.001):
    """ расчет градиента по методу наискорейшего спуска 
    Keyword args:
    func --- целевая функция, задается функцией или callable объектом
    x0 --- начальное значение кординат, список размера dims
    args --- дополнительные параметры целевой функции, словарь 
    {'имя_парамтера':значение}
    dx --- приращение по х
    Returns:
    Список с частными производными x[dim]
    """
    if not args: args = {}
    dfx = x0[:]
    f0 = func(*x0, **args)
    for dim in range(len(x0)):
        x0[dim] += dx
        f1 = func(*x0, **args)
        dfx[dim] = (f1 - f0)/dx
    return dfx

def minimize(func, x0, params, args=None, eps=0.0000001):
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
    df = [1, 1]
    k = params[:]
    x = x0[:]
    f0 = func(*x, **args)
    ff = f0+2*eps
    xs = []
    xs.append(x[:])
    while abs(f0 - ff) > eps:
        cnt += 1
        f0 = ff
        dfo = df
        df = grd(func, x, args = args)
        
        for dim in range(len(x0)):
            if sign(dfo[dim]) != sign(df[dim]):
                k[dim] /= 2
            x[dim] -= k[dim]*df[dim]

        xs.append(x[:])
        ff = func(*x, **args)
    return x, ff, xs, cnt

def func_plt(func, path):
    delta = 0.025
    x = np.arange(func['bnds'][0][0], func['bnds'][0][1], delta)
    y = np.arange(func['bnds'][1][0], func['bnds'][1][1], delta)
    X, Y = np.meshgrid(x, y)
    if func['args']:
        Z = func['func'](X, Y, **func['args'])
    else:
        Z = func['func'](X, Y)
    plt.figure()
    lc = -2*abs(func['rv'][2])/(func['rv'][0] - func['rv'][1])
    cs = plt.contour(X, Y, Z, 
                     levels = np.arange(func['rv'][0], func['rv'][1], func['rv'][2]), 
                     linewidths = np.arange(2.1, 0.1, lc), 
                     colors = 'k')
    plt.clabel(cs, inline=1, fontsize=7)

    path = np.array(path).T
    plt.plot(path[0], path[1],'b-',linewidth=2)
    plt.grid(True)
    plt.title('Gradient descent')
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
    
    args_min, min_func, path, cnt  = minimize(func = fncs[ind]['func'], 
                                              x0 = fncs[ind]['init'], 
                                              params = [5,3],
                                              args = fncs[ind]['args'])    
    
    print("min = {val}, argmin(f) = {args}, iterations = {i}".format(val=min_func,
                                                                     args=args_min, 
                                                                     i=cnt))
    func_plt(func=fncs[ind], path=path)   
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
