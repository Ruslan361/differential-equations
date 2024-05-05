import systemofDE
import sympy

def SearchInvariantLines(P, Q):
    x, y = sympy.symbols('x y')
    #y = kx + c
    #dy/dt = k dx/dt
    k, c = sympy.symbols('k c')
    Y = k * x + c
    #P = self.P.subs({self.a: a, self.b: b})
    #Q = self.Q.subs({self.a: a, self.b: b})
    
    #dy/dt - k * dx/dt
    exp = Q.subs({y: Y}) - k * P.subs({y: Y})
    
    koeffs = systemofDE.GetKoeff(exp, x, k, c)
    
    #yotx - инвариантные прямые
    yotx = []
    for coeff in koeffs:
        yotx.append((coeff[0] * x + coeff[1]))
    
    
    
    
    #x = ky + c
    #dx/dt - k * dy/dt
    X = k * y + c
    exp = P.subs({x: X}) - k * Q.subs({x: X})
    
    koeffs = systemofDE.GetKoeff(exp, y, k, c)
    
    #yotx - инвариантные прямые
    xoty = []
    for coeff in koeffs:
        xoty.append((coeff[0] * x + coeff[1]))
    #print(koeffs)
    return yotx, xoty

if __name__ == '__main__':
    x, y, a, b = sympy.symbols('x y a b')
    P = (-1) *x*(b-x-y)
    Q = (a-y)*(2*x+y)
    yotx, xoty = SearchInvariantLines(P, Q)
    print(f'yotx \n {yotx}')
    print(f'xoty \n {xoty}')