import systemofDE
import sympy
def IsMatrixDiagonalize(Matrix):
  return Matrix[0, 1] == 0 and Matrix[1, 0] == 0
  #for i, row in enumerate(Matrix):
    #for j, elem in enumerate(row):
def IsMatrixUpTriangle(Matrix):
  return Matrix[1, 0] == 0
def IsMatrixDownTriangle(Matrix):
  return Matrix[0, 1] == 0

class LatexGenerator:
  def __init__(self, P, Q):
    self.system = systemofDE.SystemOfDifferentialEquationsOnPlane(P, Q)
  def GetJacobiansInSOFE(self):
    Jacobian = self.system.GetJacobian()
    x, y, a, b = sympy.symbols('x y a b')

    sofe = self.system.GetStateOfEquilibrium()
    print(sofe)
    Jacobians = []
    for sof in sofe:
      Jacobians.append(Jacobian.subs({x: sof[x], y: sof[y]}))
    #for jac in Jacobians:
      #print(jac)
      #print(f'Matrix is diagonal - {IsMatrixDiagonalize(jac)}')
      #print(f'Matrix is up triangle - {IsMatrixUpTriangle(jac)}')
      #print(f'Matrix is down triangle - {IsMatrixDownTriangle(jac)}')
    return sofe, Jacobians


  def GetCharacteristic(self, sofes, Jacobians):
    characteristics = []
    for sof, jac in zip(sofes, Jacobians):
      characteristic = {"state": sof}
      if IsMatrixDiagonalize(jac) or IsMatrixUpTriangle(jac) or IsMatrixDownTriangle(jac):
        characteristic["special type"] = True
        characteristic["eigenvals"] = (jac[0,0], jac[1,1])
      else:
        characteristic["special type"] = False
        characteristic["jacobian"] = jac
      characteristics.append(characteristic)
    return characteristics

  def GetLambdas(self, char, an, bn):
    x, y, a, b = sympy.symbols('x y a b')
    lambdas = []
    if (char["special type"]):
      for eigenval in char["eigenvals"]:
        lambdas.append(eigenval.subs({a: an, b: bn}))
    else:
      jac = char["jacobian"]
      jac = jac.subs({a: an, b: bn})
      lambdas.extend(systemofDE.GetEigenvalues(jac))
    return lambdas

  def GetTypeOfSOFE(self, char, an, bn):
    result = 'так как '
    x, y, a, b = sympy.symbols('x y a b')
    lambdasvalues = self.GetLambdas(char, an, bn)
    if (char["special type"]):
      lambdas = char["eigenvals"]

      for i, l in enumerate(lambdas):
        result += '$'
        result += str(l)
        ln = l.subs({a: an, b: bn})
        if ln < 0:
          result += '  < 0 '
        if ln > 0:
          result += ' > 0 '
        if ln == 0:
          result += ' = 0 '
        result += '$'
        if i == 0:
          result += ' и '
    else:
      for i, l in enumerate(lambdasvalues):
        result += '${\\lambda_{'+ str(i+1) +'}}' + ' = ' + sympy.latex(l)
        if sympy.im(l) == 0:
          if l < 0:
            result += '  < 0 '
          if l > 0:
            result += ' > 0 '
          if l == 0:
            result += ' = 0 '
          result += '$'
        if i == 0:
          result += ' и '

    result += f" то $({char['state'][x].subs({a: an, b:bn})}, {char['state'][y].subs({a: an, b:bn})})$ -- " + systemofDE.GetTypeOfEq(lambdasvalues)
    if (char["special type"]):
      result = result.replace('a', 'a^\\ast')
      result = result.replace('b', 'b^\\ast')
    return result


  def GetReport(self, an, bn):
    x, y, a, b = sympy.symbols('x y a b')
    P = self.system.P.subs({a: an, b:bn})
    Q = self.system.Q.subs({a: an, b:bn})
    Ps = str(P).replace('*', ' \\cdot ')
    Qs = str(Q).replace('*', ' \\cdot ')
    latex = f"Возьмем следующие значения параметров:  $a^\\ast = {an}$, $b^\\ast = {bn}$. При этих параметрах система будет иметь следующий вид: \n\n"
    latex += "$$\n\
\\left \lbrace \n\
\\begin{matrix} \n"
    latex+= "\dot{x} = " + f"{Ps}, \\\\\n"
    latex+= "\dot{y} = " + f"{Qs}. \\\n"
    latex+= "\end{matrix} \n\
\\right . .$$\n\n"


    latex += "Состояния равновесия данной системы с выбранными параметрами: "

    sofe, Jacobians = self.GetJacobiansInSOFE()
    chars = self.GetCharacteristic(sofe, Jacobians)
    for i, char in enumerate(chars):
      latex += f"$({char['state'][x].subs({a: an, b:bn})}, {char['state'][y].subs({a: an, b:bn})})$"
      if (i < len(chars)-1):
        latex+=', '
      else:
        latex += '. '
    latex+= "Определим тип каждого состояния равновесия, проверив, какому из неравенств удовлетворяют взятые значения параметров.  Пересмотрим наши записи для каждого состояния равновесия: \n"
    latex += '\\begin{itemize}\n'
    for i, char in enumerate(chars):
      latex += "\item{ " + self.GetTypeOfSOFE(char, an, bn)
      if i == 3:
        latex += '.'
      else:
        latex += ';'
      latex+= '}\n'
    latex += "\end{itemize} \n\n"
    return latex
    #print(self.GetTypeOfSOFE(chars[0], -4, 4))




if __name__ == '__main__':
  x, y, a, b = sympy.symbols('x y a b')
  P = (-1) *x*(b-x-y)
  Q = (a-y)*(2*x+y)
  generator = LatexGenerator(P, Q)
  latex = generator.GetReport(4, 1)
  # Открываем файл для записи (если файл не существует, он будет создан)
  with open('file.txt', 'w', encoding='utf-8') as file:
    # Записываем строку в файл
    file.write(latex)
