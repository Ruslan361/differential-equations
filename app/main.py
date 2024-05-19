from random import randint
import systemofDE
from kivy.uix.stacklayout import StackLayout
from sympy.parsing.sympy_parser import parse_expr
from kivy.core import text
from sympy.core import N
from sympy import symbols, sin, S, Eq, Interval, Union
from sympy.calculus.util import continuous_domain
#from spb import line, plot, MB
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg as FCKA
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from randcolor import RandColor

#import multiprocessing
#multiprocessing.set_start_method('spawn', True)
from concurrent import futures

def plot_yx(expr, x_range=(-10, 10), label=""):
    """
    Рисует график функции y = f(x) с использованием SymPy и Matplotlib.

    Параметры:
    expr (sympy expression): Выражение для функции y в зависимости от x.
    x_range (tuple, optional): Диапазон значений x для построения графика. По умолчанию (-10, 10).
    title (str, optional): Заголовок графика.
    xlabel (str, optional): Подпись оси x.
    ylabel (str, optional): Подпись оси y.
    """
    x = sp.symbols('x')
    #f = sp.Function('f')(expr)
    f = sp.lambdify(x, expr + 0 * x, modules='sympy')

    x_vals = np.linspace(x_range[0], x_range[1], 800)
    y_vals = f(x_vals)

    if (isinstance(y_vals, int)):
        y_vals = np.full(800, y_vals)
    elif (isinstance(y_vals, float)):
        y_vals = np.full(800, y_vals)
    #plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, '--b', label=label)



def plot_xy(expr, y_range=(-10, 10), label=""):
    """
    Рисует график функции y = f(x) с использованием SymPy и Matplotlib.

    Параметры:
    expr (sympy expression): Выражение для функции y в зависимости от x.
    x_range (tuple, optional): Диапазон значений x для построения графика. По умолчанию (-10, 10).
    title (str, optional): Заголовок графика.
    xlabel (str, optional): Подпись оси x.
    ylabel (str, optional): Подпись оси y.
    """
    y = sp.symbols('y')
    f = sp.lambdify(y, expr + 0*y, modules='sympy')

    y_vals = np.linspace(y_range[0], y_range[1], 800)
    x_vals = f(y_vals)
    if (isinstance(x_vals, int)):
        x_vals = np.full(800, x_vals)
    elif (isinstance(y_vals, float)):
        x_vals = np.full(800, x_vals)
    #plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, '--b', label=label)

def new_method1(system, x, y, a, b, state, vector_len, vector_width, vec):

    dx = float (vec[0].evalf())
    dy = float (vec[1].evalf())


    return [new_method(system, x, y, a, b, state, dx, dy) , new_method(system, x, y, a, b, state, -dx, -dy)]


def new_method(system, x, y, a, b, state, dx, dy):
    evector = np.array([[dx],[dy]])

    xn = float(state[x].evalf()) + dx * 1e-3
    yn = float(state[y].evalf()) + dy * 1e-3

    dn = system.GetDirection(xn, yn, a, b)
    dxn = float(dn['P'])
    dyn = float(dn['Q'])
    nearvector = np.array([[dxn],[dyn]])

    llcosa = np.dot(evector.transpose(), nearvector)
    if llcosa > 0:
        return [GetDrawSystem(system, xn, yn, 1000000, 0.0005, mode='--r'), '--r']
    else:
        return [GetDrawSystem(system, xn, yn, 1000000, -0.0005, mode='--g'), '--g']
    
    #p = self.system.Plot(nulklines["P = 0"])
    #p.extend(self.system.Plot(nulklines["Q = 0"]))
def GetDrawSystem(system, startx: float, starty:float, steps:int, sizeofstep: float, a = 1, b = 1, mode='--k'):
    x, y = startx, starty
    xpoints = []
    ypoints = []
    P = system.P.subs({system.a: a, system.b: b})
    Q = system.Q.subs({system.a: a, system.b: b})
    Pl = sp.lambdify((system.x, system.y), P, modules='sympy')
    Ql = sp.lambdify((system.x, system.y), Q, modules='sympy')

    for i in range(steps):
        dx = Pl(x, y)
        dy = Ql(x, y)
        #d = system.GetDirection(x, y, a, b)
        #dx = float(d['P'])
        #dy = float(d['Q'])
        xpoints.append(float(x))
        ypoints.append(float(y))
        l = pow(dx**2 + dy**2, 0.5)
        if not l == 0:
            dx = dx/l
            dy = dy / l
        x = dx * sizeofstep + x
        y = dy * sizeofstep + y
    return xpoints, ypoints


def DrawSystem(system, startx: float, starty:float, steps:int, sizeofstep: float, a = 1, b = 1, mode='--k'):
    x, y = startx, starty
    xpoints = []
    ypoints = []
    P = system.P.subs({system.a: a, system.b: b}, modules='sympy')
    Q = system.Q.subs({system.a: a, system.b: b}, modules='sympy')
    Pl = sp.lambdify((system.x, system.y), P)
    Ql = sp.lambdify((system.x, system.y), Q)

    for i in range(steps):
        dx = Pl(x, y)
        dy = Ql(x, y)
        #d = system.GetDirection(x, y, a, b)
        #dx = float(d['P'])
        #dy = float(d['Q'])
        xpoints.append(x)
        ypoints.append(y)
        l = pow(dx**2 + dy**2, 0.5)
        if not l == 0:
            dx = dx / l
            dy = dy / l
        x = dx * sizeofstep + x
        y = dy * sizeofstep + y
    plt.plot(xpoints, ypoints, mode)

def arrowbuild(x: float, y: float, dx: float, dy: float, vector_len:float, vector_width, color = 'blue') -> None:
    l = (dx**2 + dy**2)**(0.5)
    if not (l == 0):
        dx = dx/l
        dy=dy/l



    dx = dx * vector_len
    dy = dy * vector_len
    #arrowprops=arrowprops(arrowstyle='<-', color='blue', linewidth=10, mutation_scale=150)
    plt.arrow(x=x, y=y, dx=dx, dy=dy, shape='full', width= vector_width, color=color)


def build(graphics, left_border_x, right_border_x, left_border_y, right_border_y, count):
    # Определяем уравнение
    x = sp.symbols('x')
    y = sp.symbols('y')
    plt.clf()
    for left, right in graphics:





        #func = x**2 + y**2
        equation = Eq(left - right, 0)
        x_values = np.linspace(left_border_x, right_border_x, count)  # Пример: от -1.5 до 1.5 с шагом 400
        y_values = np.linspace(left_border_y, right_border_y, count)  # Пример: от -1.5 до 1.5 с шагом 400
        Z = sp.lambdify((x, y), (left - right), 'numpy')

        Xm, Ym = np.meshgrid(x_values, y_values)
        label_g = f"{str(left)} = {str(right)}"

        #plt.figure()
        color = RandColor()
        plt.contour(Xm, Ym, Z(Xm, Ym), [0], colors=color, show=False, label=label_g)
        #line = mlines.Line2D([], [], color=color, label=label_g)
        plt.scatter(-100000, 1000000, color=color, marker='o', label=label_g)
        plt.legend()
        plt.xlim(left_border_x, right_border_x)
        plt.ylim(left_border_y, right_border_y)




    plt.grid(True)
    #plt.show()
        #plt.show()


        # Convert the Sympy plot to a Kivy compatible format
        #fig = sympy_plot._fig
    plt.legend()
    #graph = FCKA(figure=plt.gcf())
    #return graph

import gc
class Container(StackLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        x = sp.symbols('x')
        y = sp.symbols('y')
        left = x**2 + y**2
        right = 1

        build([(left, right)], -2, 2, -2, 2, 400 )
        self.graph = FCKA(figure=plt.gcf())
        self.graph.size = (500, 500)
        self.graph.size_hint = (None, None)
        self.scroll.add_widget(self.graph)
    def GraphRefresh(self):
        self.scroll.remove_widget(self.graph)
        self.graph.clear_widgets()
        del self.graph

        gc.collect()
        self.graph = FCKA(figure=plt.gcf())
        self.scroll.add_widget(self.graph)
        self.graph.mpl_connect("button_press_event", self.on_graph_click)
        self.graph.size = (1000, 1000)
        self.graph.size_hint = (None, None)

    def analiz(self):
        plt.close()
        result = ""

        x = sp.symbols('x')
        y = sp.symbols('y')
        a = int(self.texta.text)
        b = int(self.textb.text)
        local_dict = {'x': sp.symbols('x'), 'y': sp.symbols('y'), 'a': a, 'b': b}
        P = parse_expr(self.textdxdt.text, local_dict)
        Q = parse_expr(self.textdydt.text, local_dict)
        self.system = systemofDE.SystemOfDifferentialEquationsOnPlane(P, Q)



        result += "Поиск состояний равновесия и их анализ \n"
        SofE = self.system.GetStateOfEquilibriumAB(a, b)
        characteristics = self.system.GetCharacteristicOfStateOfEquilibriumAB(SofE, a, b)



        for char in characteristics:
            for key in char:
                result += str(key) + ':' + str(char[key]) + '\n\n\n'



        result += ("Поиск уравнений нульклин \n")
        nulklines = self.system.GetNulklines(a, b)

        result += str(nulklines) + '\n'
        result += ("Изображение нульклин") +'\n'

        lines = []
        for key in nulklines:
            for first in nulklines[key]:
                for first_f in first:
                    lines.append((first_f, first[first_f]))


        left_border_x = float(self.leftborderx.text)
        right_border_x = float(self.rightborderx.text)
        left_border_y = float(self.leftbordery.text)
        right_border_y = float(self.rightbordery.text)
        count = int(self.count.text)

        build(lines, left_border_x, right_border_x, left_border_y, right_border_y , 800)
        for char in characteristics:
            state = char["Состояние равновесия"]
            vectors = char["собственные вектора"]
            vector_len = float(self.vec_len.text)
            vector_width = float(self.vec_width.text)

            self.DrawAsync(x, y, a, b, state, vectors, vector_len, vector_width)

            states_color = {'седло': 'red', 'касп': 'yellow', 'устойчивый узел': 'green', 'неустойчивый узел': 'green',  'неустойчивый фокус': 'yellow', 'устойчивый фокус': 'yellow'}
            SOFEType = char["тип состояния равновесия"]
            if SOFEType in states_color:
                plt.scatter(float(state[x].evalf()), float(state[y].evalf()), color=states_color[SOFEType], marker='o', label=SOFEType, s=vector_width*1000)


        print(self.system.SearchInvariantLines(a, b))

        plt.legend()

        yotx, xoty = self.system.SearchInvariantLines(a, b)
        for ys in yotx:
            plot_yx(ys, (left_border_x, right_border_x), str(ys))
        for xs in xoty:
            plot_xy(xs, (left_border_y, right_border_y), str(xs))
        self.GraphRefresh()
        lines  = self.system.SearchInvariantLines(a, b)
        result += " инвариантные прямые \n"
        for line  in lines[0]:
          result+= "y = " + str(line) + '\n'
        for line in lines[1]:
          result += "x = " + str(line) + '\n'

        self.report.text = result

    def DrawAsync(self, x, y, a, b, state, vectors, vector_len, vector_width):
        with futures.ProcessPoolExecutor(max_workers=12) as executor:
            todo = []
            for vec in vectors:
                    #print(new_method1(self.system,x, y, a, b, state, vector_len, vector_width, vec))
                if sp.im(vec[0]) == 0 and sp.im(vec[1]) == 0:
                    future = executor.submit(new_method1, self.system, x, y, a, b, state, vector_len, vector_width, vec)
                    todo.append(future)
                    dx = float (vec[0].evalf())
                    dy = float (vec[1].evalf())
                    arrowbuild(float(state[x].evalf()), float(state[y].evalf()), dx, dy, vector_len, vector_width, 'red')
                    #self.new_method1(x, y, a, b, state, vector_len, vector_width, vec)
            for future in futures.as_completed(todo):
                    #print()
                r = future.result()
                plt.plot(r[0][0][0], r[0][0][1], r[0][1])
                plt.plot(r[1][0][0], r[1][0][1], r[1][1])
                r.clear()



    def on_graph_click(self, event):
        # Обработка события нажатия на график
        if event.button == 3:
            x, y = event.xdata, event.ydata
            if not x == None and not y == None:
                DrawSystem(self.system, x, y, 10000000, 0.00005)
                self.GraphRefresh()
        if event.button == 1:
            x, y = event.xdata, event.ydata
            #Ха-ха костыль
            if not x == None and not y == None:
                d = self.system.GetDirection(x, y, 1, 1)
                dx = float(d['P'])
                dy = float(d['Q'])

                vector_len = float(self.vec_len.text)
                vector_width = float(self.vec_width.text)

                arrowbuild(x, y, dx, dy, vector_len, vector_width)
                self.GraphRefresh()

                #print(f"Clicked on point: x = {x}, y = {y}")


class SympyPlotApp(App):
    def build(self):
        container = Container()
        return container


if __name__ == "__main__":
    SympyPlotApp().run()

"""
    x, y, a, b = sympy.symbols('x y a b')
    a, b = sympy.symbols('a b', real=True)
    P = (-1) *x*(b-x-y)
    Q = (a-y)*(2*x+y)

    self.system = self.systemOfDifferentialEquationsOnPlane(P, Q)
    SofE = self.system.GetStateOfEquilibriumAB(1, 2)
    self.system.GetNulklines(1,2)

    self.system.GetJacobian()
    characteristics = self.system.GetCharacteristicOfStateOfEquilibriumAB(SofE, 1, 2)
    for char in characteristics:
        print(char)
    nulklines =self.system.GetNulklines(1, 2)
    print(nulklines)
    p = self.system.Plot(nulklines)
    p.show()
"""
"""
    print(type(P))
    #состояния равновесия
    eq1 = sympy.Eq(P, 0)
    print(eq1)

    eq2 = sympy.Eq(Q, 0)
    print(eq2)
    solution = sympy.solve([eq1,eq2], [x, y], dict=True)
    print(solution)


    Px = sympy.diff(P, x)
    Py = sympy.diff(P, y)
    Qx = sympy.diff(Q, x)
    Qy = sympy.diff(Q, y)

    print("enter a, b")
    an = (float)(input())
    bn = (float)(input())

    #Отоброжаем нульклины
    Pp = P.subs(a, an)
    Pp = Pp.subs(b, bn)
    Pp = sympy.Eq(Pp, 0)
    print(type(Pp))
    print(Pp)
    nulkl = sympy.solve(Pp, [x,y], dict=True)
    print(nulkl)

    Qp = Q.subs(a, an)
    Qp = Qp.subs(b, bn)
    Qp = sympy.Eq(Qp, 0)
    print(Qp)
    nulkl.extend(sympy.solve(Qp, [x,y]))
    print(nulkl)
    p1 = plot(xlim=[-100,100], ylim=[-100,100], aspect_ratio=(1, 1), show =False, adaptive=True, color="grey")
    #p1 = plot(nulkl[0], show=True)
    print(type(p1))
    print()
    print()
    print("Plotting nullklines...")
    #for n in nulkl:
    #    print("y = ", n)
    #    p2 = plot(n, (x, -100, 100), xlim=[-100,100], ylim=[-1000,1000], aspect_ratio=(1, 1), autoscale=True, show=False)

    #    p1.extend(p2)
    #for pl in p1:
    #    pl.line_aspect = (1, 1)
    p = plot(Pp, (x, -100, 100), xlim=[-100,100], ylim=[-100,100], aspect_ratio=(1, 1), autoscale=True, show=False)
    p1.extend(p)
    p = plot(Qp, (x, -100, 100), xlim=[-100,100], ylim=[-1000,1000], aspect_ratio=(1, 1), autoscale=True, show=False)
    p1.extend(p)

    #p = sympy.plot_implicit(Pp, (x, -100, 100), (y, -100, 100), points=100000, show=False)
    #p1.extend(p)
   # p = sympy.plot_implicit(Qp, (x, -100, 100), (y, -100, 100), points=100000, show=False)
   #p1.extend(p)
    p1.show()
"""



#if __name__ == "__main__":
#    main()
