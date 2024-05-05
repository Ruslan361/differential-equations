import sympy
#from latex2sympy2 import latex2sympy, latex2latex
from sympy.plotting import plot
#import matplotlib.pyplot as plt
import threading
from randcolor import RandColor

#Поиск коэффициэнтов методом неопределённых коэффициэнтов
def GetKoeff(exp, variable, k, c) -> tuple:
    #expr = expr.subs({variable, sympy.symbols('x')})
    exp = sympy.Poly(exp, variable)
    coeff = exp.coeffs()
    eq = []
    for co in coeff:
        eq.append(sympy.Eq(co, 0))
    result = sympy.solve(eq, (k, c))
    return result

def GetEigenvalues(Matrix):
    eigenvalues = list()
    current = Matrix.eigenvals()
    for key in current:
        for i in range(current[key]):
            eigenvalues.append(key)
    return eigenvalues

def show_thread(system, a, b):
    while (True):
        print("enter x, y")
        xn = (float)(input())
        yn = (float)(input())
        direction = system.GetDirection(xn, yn, a, b)
        if direction['P'] < 0:
            print("left")
        elif direction['P'] > 0:
            print("right")
        else:
            print("0")
        if direction['Q'] < 0:
            print("down")
        elif direction['Q'] > 0:
            print("up")
        else:
            print("0")
        print(direction)

class SystemOfDifferentialEquationsOnPlane:
    def __init__(self, P: sympy.core.mul.Mul , Q: sympy.core.mul.Mul):
        self.P = P
        self.Q = Q
        self.x, self.y, self.a, self.b = sympy.symbols('x y a b')
        self.a, self.b = sympy.symbols('a b', real=False)
    def GetStateOfEquilibrium(self) -> dict:
        eq1 = sympy.Eq(self.P, 0)
        #print(eq1)
        eq2 = sympy.Eq(self.Q, 0)
        #print(eq2)
        solution = sympy.solve([eq1,eq2], [self.x, self.y], dict=True)
        return solution
    def GetStateOfEquilibriumAB(self, a: float, b: float):
        solution = self.GetStateOfEquilibrium()
        #print("\n\n\n\n")
        result = list()
        #print(type(solution))
        for sol in solution:
            #print(sol)
            current = dict()
            current[self.x] = sol[self.x].subs({self.a: a, self.b: b})
            current[self.y] = sol[self.y].subs({self.a: a, self.b: b})
            result.append(current)
            #print(result)
        return result
    def GetNulklines(self, a:float, b:float):
        Pp = self.P.subs({self.a: a, self.b: b})
        PEq = sympy.Eq(Pp, 0)
        nulkl = dict()
        Qp = self.Q.subs({self.a: a, self.b: b})
        QEq = sympy.Eq(Qp, 0)
        nulkl["P = 0"] = sympy.solve(PEq, [self.x,self.y], dict=True)
        nulkl["Q = 0"] = sympy.solve(QEq, [self.x,self.y], dict=True)
        return nulkl
    def Plot(self, functions):
        p1 = plot(xlim=[-100,100], ylim=[-100,100], aspect_ratio=(1, 1), show =False, adaptive=True, block=False)
        for func in functions:
            if self.y in func:
                p = plot(func[self.y], (self.x, -100, 100), xlim=[-100,100], ylim=[-100,100], aspect_ratio=(1, 1), autoscale=True, show=False, block=False)
                p1.extend(p)
            elif self.x in func:
                eq = sympy.Eq(func[self.x], self.x)
                if func[self.x].has(self.y):
                    yotx = sympy.solve(eq, [self.y], dict=True, block=False)
                    if yotx:
                        #functions.remove(func)
                        functions.extend(yotx)
                        continue
                              
                p = sympy.plot_implicit(eq, (self.x, -100, 100), (self.y, -100, 100), points=100000000, show=False, block=False)
                p1.extend(p)
            else:
                raise Exception("wrong input for plot")
        return p1
    def GetJacobian(self):
        Px = sympy.diff(self.P, self.x)
        Py = sympy.diff(self.P, self.y)
        Qx = sympy.diff(self.Q, self.x)
        Qy = sympy.diff(self.Q, self.y)
        J = sympy.Matrix([[Px,Py], [Qx, Qy]])
        return J
    def GetDirection(self, x:float, y: float, a:float, b:float) -> dict:
        P = self.P.subs({self.x: x, self.y: y, self.a: a, self.b: b})
        Q = self.Q.subs({self.x: x, self.y: y, self.a: a, self.b: b})
        #P = sympy.simplify(P)
        #Q = sympy.simplify(Q)
        return {"P": P, "Q": Q}
    
    def GetEigenVectorAndValues (self, Matrix) -> tuple:
        eigenvectors = list()
        eigenvalues = list()
        #Matrix = sympy.Matrix([[1, 0], [0,1]])
        current = Matrix.eigenvects()
        for elem in current:
            for i in range(elem[1]):
                eigenvalues.append(elem[0])
            for e in elem[2]:
                eigenvectors.append(e)
                
        return eigenvalues, eigenvectors
    def Linearization(self, x0, y0, a, b):
        Px = sympy.diff(self.P, self.x).subs({self.x: x0, self.y: y0, self.a: a, self.b: b})
        Py = sympy.diff(self.P, self.y).subs({self.x: x0, self.y: y0, self.a: a, self.b: b})
        Qx = sympy.diff(self.Q, self.x).subs({self.x: x0, self.y: y0, self.a: a, self.b: b})
        Qy = sympy.diff(self.Q, self.y).subs({self.x: x0, self.y: y0, self.a: a, self.b: b})
        
        dxdt = Px * (self.x - x0) + Py * (self.y - y0)
        dydt = Qy * (self.x - x0) + Qy * (self.y - y0)
        return dxdt, dydt
        
    def GetLinearizationEigenVectors(self, x0, y0, a, b):
        dxdt, dydt = self.Linearization(x0, y0, a, b)
        Px = sympy.diff(dxdt, self.x)
        Py = sympy.diff(dxdt, self.y)
        Qx = sympy.diff(dydt, self.x)
        Qy = sympy.diff(dydt, self.y)
        J = sympy.Matrix([[Px,Py], [Qx, Qy]])
        return list(zip(dxdt, dydt, self.GetEigenVectorAndValues(J)))
        
    def GetCharacteristicOfStateOfEquilibriumAB(self, SofE:list, a:float, b:float) -> list:
        J = self.GetJacobian()
        J = J.subs({self.a: a, self.b: b})

        result = list()
        for state in SofE:
            JatState = J.subs({self.x: state[self.x], self.y: state[self.y]})
            #print()
            #print(state)
            #print(JatState)
            #print()
            eigenvalues, eigenvectors = self.GetEigenVectorAndValues(JatState)

            
            if len(eigenvalues) == 2:
                Re_0 = sympy.re(eigenvalues[0])
                Im_0 = sympy.im(eigenvalues[0])
                Re_1 = sympy.re(eigenvalues[1])
                Im_1 = sympy.im(eigenvalues[1])
                if Re_0 == 0 or Re_1 == 0:
                    #Проверка на сложный фокус или центр
                    if Im_1 == Im_0 and (not Im_1 == 0):
                        result.append({"Состояние равновесия": state, "собственные числа": eigenvalues, "собственные вектора": eigenvectors, "тип состояния равновесия": "сложный фокус или центр"})
                    #Проверка на касп
                    elif Im_1 == Im_0 and Im_1 == 0:
                        result.append({"Состояние равновесия": state, "собственные числа": eigenvalues,"собственные вектора": eigenvectors, "тип состояния равновесия": "Касп"})
                    elif Im_1 == 0 or Im_0 == 0:
                        result.append({"Состояние равновесия": state, "собственные числа": eigenvalues,"собственные вектора": eigenvectors, "тип состояния равновесия": "Седлоузел"})
                    else:
                        result.append({"Состояние равновесия": state, "собственные числа": eigenvalues,"собственные вектора": eigenvectors, "тип состояния равновесия": "Негрубое состояние равновесия"})
                else:
                    #Проверка на вещественность числа
                    if Im_0 == 0 and Im_1 == 0:
                        #проверка на седло
                        if Re_0 > 0 and Re_1 < 0 or Re_0 < 0 and Re_1 > 0:
                            result.append({"Состояние равновесия": state, "собственные числа": eigenvalues,"собственные вектора": eigenvectors, "тип состояния равновесия": "Седло"})
                        #elif Re_0 != Re_1:
                        elif Re_0 > 0 and Re_1 > 0:
                            result.append({"Состояние равновесия": state, "собственные числа": eigenvalues,"собственные вектора": eigenvectors, "тип состояния равновесия": "Неустойчивый узел"})
                        else:
                            result.append({"Состояние равновесия": state, "собственные числа": eigenvalues,"собственные вектора": eigenvectors, "тип состояния равновесия": "Устойчивый узел"})
                    #else:
                           # result.append({"Состояние равновесия": state, "собственные числа": eigenvalues, "тип состояния равновесия": "Нужны дополнительные исследования"})
                    else:
                        #Число комплексное
                        if Re_0 > 0 and Re_1 > 0:
                            result.append({"Состояние равновесия": state, "собственные числа": eigenvalues,"собственные вектора": eigenvectors, "тип состояния равновесия": "Неустойчивый фокус"})
                        elif Re_0 < 0 and Re_1 < 0:
                            result.append({"Состояние равновесия": state, "собственные числа": eigenvalues,"собственные вектора": eigenvectors, "тип состояния равновесия": "Устойчивый фокус"}) 
                        else:
                            result.append({"Состояние равновесия": state, "собственные числа": eigenvalues,"собственные вектора": eigenvectors, "тип состояния равновесия": "Нужны дополнительные исследования"}) 
        return result
    def SearchInvariantLines(self, a, b):
        #y = kx + c
        #dy/dt = k dx/dt
        k, c = sympy.symbols('k c')
        Y = k * self.x + c
        P = self.P.subs({self.a: a, self.b: b})
        Q = self.Q.subs({self.a: a, self.b: b})
        
        #dy/dt - k * dx/dt
        exp = Q.subs({self.y: Y}) - k * P.subs({self.y: Y})
        
        koeffs = GetKoeff(exp, self.x, k, c)
        
        #yotx - инвариантные прямые
        yotx = []
        for coeff in koeffs:
            yotx.append((coeff[0] * self.x + coeff[1]))
        
        
        
        
        #x = ky + c
        #dx/dt - k * dy/dt
        X = k * self.y + c
        exp = P.subs({self.x: X}) - k * Q.subs({self.x: X})
        
        koeffs = GetKoeff(exp, self.y, k, c)
        
        #yotx - инвариантные прямые
        xoty = []
        for coeff in koeffs:
            xoty.append((coeff[0] * self.x + coeff[1]))
        #print(koeffs)
        return yotx, xoty

    