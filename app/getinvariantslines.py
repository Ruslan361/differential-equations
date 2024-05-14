import systemofDE
import sympy

def SearchInvariantLines(P, Q):
    x, y = sympy.symbols('x y')
    #y = kx + c
    #dy/dt = k dx/dt
    k, c = sympy.symbols('k c')
    a, b = sympy.symbols('a b')
    Y = k * x + c
    #P = self.P.subs({self.a: a, self.b: b})
    #Q = self.Q.subs({self.a: a, self.b: b})

    #dy/dt - k * dx/dt
    exp = Q.subs({y: Y}) - k * P.subs({y: Y})

    koeffs = systemofDE.GetKoeffAB(exp, x, k, c, a, b)

    #yotx - инвариантные прямые
    yotx = []
    for coeff in koeffs:
        yotx.append([(coeff[0] * x + coeff[1]), {'a': coeff[2], 'b': coeff[3]}])




    #x = ky + c
    #dx/dt - k * dy/dt
    X = k * y + c
    exp = P.subs({x: X}) - k * Q.subs({x: X})

    koeffs = systemofDE.GetKoeffAB(exp, y, k, c, a, b)

    #yotx - инвариантные прямые
    xoty = []
    for coeff in koeffs:
        xoty.append([(coeff[0] * y + coeff[1]), {'a': coeff[2], 'b': coeff[3]}])
    #print(koeffs)
    return yotx, xoty

if __name__ == '__main__':



    x, y, a, b = sympy.symbols('x y a b')
    P = (-1) *x*(b-x-y)
    Q = (a-y)*(2*x+y)
    yotx, xoty = SearchInvariantLines(P, Q)
    for yx in yotx:
        print(f'y = {yx[0]}, при условии a = {yx[1]['a']} и b = {yx[1]['b']}')
    print('\n\n')
    for xy in xoty:
        print(f'x = {xy[0]}, при условии a = {xy[1]['a']} и b = {xy[1]['b']}')
    #print(f'yotx \n {yotx}')
    #print(f'xoty \n {xoty}')
