import systemofDE
import re
from os import path, listdir
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
        characteristic["jacobian"] = jac
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
        result += '${\\lambda_{'+ str(i+1) +'}}' + ' = ' + sympy.latex(l) + "$"
        if sympy.im(l) == 0:
          if l < 0:
            result += '$  < 0 $'
          if l > 0:
            result += '$ > 0 $'
          if l == 0:
            result += '$ = 0 $'
          #result += '$'
        if i == 0:
          result += ' и '
    typeofSOFE = systemofDE.GetTypeOfEq(lambdasvalues)
    result += f" то $({char['state'][x].subs({a: an, b:bn})}, {char['state'][y].subs({a: an, b:bn})})$ -- " + typeofSOFE
    if (char["special type"]):
      result = result.replace('a', 'a^\\ast')
      result = result.replace('b', 'b^\\ast')
    eigenvalues, eigenvectors = systemofDE.GetEigenVectorAndValues(char["jacobian"].subs({a: an, b:bn}))
    if 'узел' in typeofSOFE:
      result+= f". Найдём ведущее направление данного узла. Якобиан в состоянии равновесия $({(char["state"][x])}, {(char["state"][y])})$ выглядит следующим образом:\n\n"
      result += '$$'+ str(sympy.latex(char["jacobian"].subs({a: an, b:bn}))).replace("matrix", "pmatrix").replace("\\left[", '').replace("\\right]", '') +'. $$'

      result += f"Её собственные числа это $\lambda_1={eigenvalues[0]}$, "+ \
      "которому соотвествует вектор $\Vec{"+"V_1}=$ $"+ sympy.latex(eigenvectors[0])+'$' + \
      f" и $\lambda_2={eigenvalues[1]}$, "+"которому соотвествует вектор $\Vec"+"{"+"V_2}=$$"+ \
      sympy.latex(eigenvectors[1])+f"$."
      if abs(eigenvalues[0].evalf()) < abs(eigenvalues[1].evalf()):
        mindex = 0
      else :
        mindex = 1
      result += f"Так как $\lambda_{mindex+1}={eigenvalues[mindex]}$ ближайшее к $0$ собственное число, то при подходе к состоянию равновесия траектории будут стремиться к направлению, задаваемому вектором "+"$\Vec{"+f"V_{mindex+1}"+"}=$ $"+ sympy.latex(eigenvectors[mindex]) +"$"
      # result += "В данном случае произошла транскритическая бифуркация. Выпишем всё, что потребуется при построении фазового портрета:"
    return result


  def GetReport(self, an, bn):
    x, y, a, b = sympy.symbols('x y a b')
    P = self.system.P.subs({a: an, b:bn})
    Q = self.system.Q.subs({a: an, b:bn})
    Ps = str(P).replace('*', ' \\cdot ')
    Qs = str(Q).replace('*', ' \\cdot ')
    latex = f"Выберем точку на бифуркационной диаграмме $({an}, {bn})$. Она соответствует следующим значениям параметров:  $a^\\ast = {an}$, $b^\\ast = {bn}$. При этих параметрах система будет иметь следующий вид: \n\n"
    latex += "$$\n\
\\left \lbrace \n\
\\begin{matrix} \n"
    latex+= "\dot{x} = " + f"{Ps}, \\\\\n"
    latex+= "\dot{y} = " + f"{Qs}. \\\n"
    latex+= "\end{matrix} \n\
\\right . .$$\n\n"


    latex += "Состояния равновесия данной системы: "

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
      latex += "\\item{ " + self.GetTypeOfSOFE(char, an, bn)
       
      if i == 3:
        latex += '.'
      else:
        latex += ';'
      latex+= '}\n'
    latex += "\end{itemize} \n\n"


    nulklines = self.system.GetNulklines(an, bn)
    latex += "Выпишем всё, что потребуется при построении фазового портрета:\n\n"
    latex += "Уравнения нульклин имеют следующий вид: \n\n"
    for key in nulklines:
        for first in nulklines[key]:
            for first_f in first:
                  latex += "$$"+ sympy.latex(first_f) + "=" +  sympy.latex(first[first_f]) + "$$\n"

    lines = self.system.SearchInvariantLines(an, bn)
    result = "\n\n"
    result += "Уравнения инвариантных прямых имеют следующий вид: \n"
    for line  in lines[0]:
      result+= "$$y = " + sympy.latex(line.subs({a: an, b:bn})) + '$$\n'
    for line in lines[1]:
      result += "$$x = " + sympy.latex(line.subs({a: an, b:bn})) + '$$\n'
    latex += result
    return latex
    #print(self.GetTypeOfSOFE(chars[0], -4, 4))




if __name__ == '__main__':
  x, y, a, b = sympy.symbols('x y a b')
  P = (-1) *x*(b-x-y)
  Q = (a-y)*(2*x+y)

  generator = LatexGenerator(P, Q)
  generator.system.a = a
  generator.system.b = b
  latex = generator.GetReport(0, -9)
  # Открываем файл для записи (если файл не существует, он будет создан)
  with open('file.txt', 'w', encoding='utf-8') as file:
    # Записываем строку в файл
    file.write(latex)


  currentarea = 4

  MegaReport = ""

  folder = 'phptr'
  files = []
  for name in listdir(folder):
      full_name = path.join(folder, name)
      
      if path.isfile(full_name):
          name_, _ext = path.splitext(name)

          file = {
              'каталог': folder,
              'файл': full_name,
              'файл_имя': name_,
              'файл_расширение': _ext,
          }
          files.append(file)
          print('\n'.join('{:<30} : {}'.format(*f) for f in sorted(file.items())), '\n')
  print(len(files))
  for file in files:
    #if file['файл_расширение'] == '.jpg':
      r = re.search("(-?[0-9])", file['файл_имя'])
      #print(x)
      MegaReport+="\n\n\\subsection{Область " + str(currentarea) +"}\n\n"
      MegaReport+= generator.GetReport(int(r.group(0)), int(r.group(1)))
      

      MegaReport+="Построим фазовый портрет (Рис. \\ref{fig:phportr"+str(currentarea)+"}).\n\n\
\\begin{figure}[h]\n\n\
	\includegraphics[width=\\textwidth]{"+f"{file["файл"]}"+"}\n\
	\centering\n\
	\caption{\label{fig:" + f"phportr{currentarea}" + "} Фазовый портрет системы с параметрами " + f"$a^\\ast = {str(int(r.group(0)))}$, $b^\\ast = {str(int(r.group(1)))}$"+ ".}\
\n\n\\end{figure}\
"

      print(f"Generate report point a = {r.group(0)}, b = {r.group(1)}")
      currentarea += 1
  with open('MegaReport.txt', 'w', encoding='utf-8') as file:
    # Записываем строку в файл
    file.write(MegaReport)