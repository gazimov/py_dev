import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams['figure.figsize'] = 12, 12
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

def fx1(verts,x,y):
    return -sum([1/(1+(x-v[0])**2+(y-v[1])**2) for v in verts])

def fx2(prm,x,y):
    return prm[0]*x**2 + prm[1]*y**2 + prm[2]*x*y + prm[3]*y + prm[4]*np.sin(y) 

def axis_dih(func, dim, x0, bounds, eps=0.001):
    """ Реализует метод поиска минимума вдоль одной координаты методом дихотомии.
    
    Keyword args:
    func --- целевая функция, задается функцией или callable объектом
    dim --- индекс координаты, по которой минимизируется func
    x0 --- начальное значение кординат в виде списка
    bounds --- границы поиска по каждой координате с индексом dim 
    eps --- точность поиска
    
    Returns:
    Список с обновленной координатой x[dim]
    """
    y1, y2 = bounds
    x = x0[:]
    while abs(y1-y2)>2*eps:
        s = (y1+y2)/2
        x[dim] = s - eps
        f1 = func['func'](func['args'],*x)
        x[dim] = s + eps
        f2 = func['func'](func['args'],*x)
        if (f1>f2):
            y1 = s
        else:
            y2 = s
    x[dim] = (y1 + y2)/2
    return x
        
def minimize(func, dims, x0, eps=0.0000001):
    """ Реализует метод поиска минимума покоординатным спуском.
    
    Keyword args:
    func --- целевая функция, задается функцией или callable объектом
    dims --- размерность функции
    x0 --- начальное значение кординат, список размера dims
    bounds --- границы поиска по каждой координате, список кортежей размера dims, 
    каждый кортеж --- пара значений, верхняя и нижняя граница
    eps --- точность поиска
    
    Returns:
    Кортеж из трех элементов: найденное минимизирующее значение аргументов в виде списка 
    размером dims, траектория поиска, и количество итераций
    """
    cnt = 0
    f0 = func['func'](func['args'],*x0)
    ff = f0+2*eps
    x = x0[:] 
    xs = []
    xs.append(x)
    while abs(f0 - ff) > eps:
        f0 = ff
        cnt += 1
        for dim in range(dims):
            x = axis_dih(func, dim, x, func['bnds'][dim])
            xs.append(x)
        ff = func['func'](func['args'],*x)
    return x, xs, cnt

def func_plt(func,path):
    delta = 0.025
    x = np.arange(func['bnds'][0][0], func['bnds'][0][1], delta)
    y = np.arange(func['bnds'][1][0], func['bnds'][1][1], delta)
    X, Y = np.meshgrid(x, y)
    Z = func['func'](func['args'],X,Y)
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
    plt.title('Coordinatewise descent')
    plt.show()

def main():   
    fnc = []
    fnc.append({'func': fx1, 'args': [(5,3),(7,5),(15,15)], 'bnds': [(0.0, 20.0), (0.0, 20.0)], 'rv': [0,-1.2,-0.05]})
    fnc.append({'func': fx2, 'args': [2,5,4,-3,100], 'bnds': [(-10.0, 10.0), (-10.0, 10.0)], 'rv': [300,-100,-10]})
    ind = 0

    args_min, path, cnt  = minimize(func=fnc[ind], dims=2, x0=[10, 1])    
    print("min = {val}, argmin(f) = {args}, iterations = {i}".format(val=fnc[ind]['func'](fnc[ind]['args'],*args_min),args=args_min, i=cnt))
    func_plt(func=fnc[ind],path=path)   

if __name__ == "__main__":
    main()
