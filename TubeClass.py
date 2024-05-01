import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
import matplotlib.pyplot as plt
import sympy as sym
from sympy.solvers import ode
import math


def addQdict_maker(X0, Plist, Sigma):
  ans = []
  temp = ('x0', 'sigma', 'P')
  for i in range(len(X0)):
    ans.append(dict(zip(temp, (X0[i], Sigma[i], Plist[i]))))
  return ans


class Tube:
    materials_dict = {1: {'name': "Сталь", 'rho': 7800, 'E': 2.06 * 10**11},
                    2: {'name': "Каучук", 'rho': 920, 'E': 8 * 10**6},
                    3: {'name': "Алюминий", 'rho': 2700, 'E': 0.7 * 10**11}}

    def __init__(self, l=20, R=0.5, material_num=1, thickness=0.01, N=1e5, phi=0):
        if l != 0:
            self.__l = abs(l)
        else:
            raise ValueError("L can't be 0")
        if R != 0:
            self.__R = abs(R)
        else:
            raise ValueError("R can't be 0")
        if material_num in self.materials_dict.keys():
            self.__material_num = material_num
        else:
            raise ValueError('Wrong material_num')
        if thickness != 0:
            self.__thickness = abs(thickness)
        else:
            raise ValueError("Thickness can't be 0")
        self.__N = N
        self.__phi = phi
        self.__material_num = material_num

    @property
    def l(self):  # длина трубы
        return self.__l

    @l.setter
    def l(self, l):
        if l != 0:
            self.__l = abs(l)
        else:
            raise ValueError("L can't be 0")

    @property
    def R(self):  # радиус трубы в сечении
        return self.__R

    @R.setter
    def R(self, R):
        if R != 0:
            self.__R = abs(R)
        else:
            raise ValueError("R can't be 0")
    @property
    def thickness(self):
      return self.__thickness
    @thickness.setter
    def thickness(self,thickness):
      self.__thickness = thickness
    @property
    def material_num(self):
      return self.__material_num
    @material_num.setter
    def material_num(self,material_num):
      if material_num in self.materials_dict.keys():
            self.__material_num = material_num
      else:
            raise ValueError('Wrong material_num')
    @property
    def N(self):
      return self.__N
    @N.setter
    def N(self,N):
      self.__N = N
    @property
    def phi(self):
      return self.__phi
    @phi.setter
    def phi(self,phi):
      self.__phi = phi
    @property
    def E_soil(self):
      return 4 * 10 ** 7 #модуль Юнга почвы
    @property
    def rho(self):
      return self.materials_dict[self.material_num]['rho']
    @property
    def E(self):
      return self.materials_dict[self.material_num]['E']
    @property
    def m(self):
      return self.rho * self.l * np.pi * (self.R**2 - (self.R-self.thickness)**2)
    def q(self): # этот метод будет изменен в трубе
      return self.m * 9.8 / self.l #погонный вес трубы
    def k(self):
      return 40 * 10**6 * 2 * np.sqrt(self.q * (1 - 0.09) * self.R / np.pi / self.E_soil)
    @property
    def Jx(self):
      return np.pi * self.R**4 / 64 * (1 - (self.R-self.thickness)**4 / self.R**4)
class TubeSolver(Tube):
  # для добавления кейсов надо править dict_id_dimensions, fun_maker, bc_maker
  dict_id_dimensions = {1:4,2:4}
  bc_id_dimensions = {0:2,1:4,2:4,3:4,4:4,5:4}
  def __init__(self,l=20,R=0.5,material_num=1,thickness=0.01,N=1e5,phi=0,task_id=1,bc_id=1,totalNodes=1000,addQGauss_params=[],addQ_params=[],Q_on=True):
    super().__init__(l=l,R=R,material_num=material_num,thickness=thickness,N=N,phi=phi)
    if task_id in self.dict_id_dimensions.keys():
      self.__task_id = task_id
    else:
      raise ValueError('Wrong task_id')
    if bc_id in self.bc_id_dimensions.keys() and self.bc_id_dimensions[bc_id] == self.dict_id_dimensions[self.task_id]:
      self.__bc_id = bc_id
    else:
      print('bc_id and task_id doesnt match. Set it to default for this task_id')
      self.__bc_id = 1 if self.dict_id_dimensions[self.task_id] == 4 else 0
    if totalNodes!=0:
      self.__totalNodes = abs(totalNodes)
    else:
      raise ValueError("totalNodes can't be 0")
    if type(addQGauss_params) is not list:
      self.__addQGauss_params = [{'x0': 0, 'sigma' : 0, 'P' : 0}]
      print('addQGauss_params didnt set correct .Made it zero')
    else:
      self.__addQGauss_params = addQGauss_params
    if type(addQ_params) is not list:
      self.__addQ_params = [{'x0': 0, 'sigma' : 0, 'P' : 0}]
      print('addQ_params didnt set correct .Made it zero')
    else:
      self.__addQ_params = addQ_params
    if type(Q_on) is not bool:
      self.__Q_on = True
      print('Q_on didnt set correct. Made it True')
    else:
      self.__Q_on = Q_on
  def fun_maker(self):
    dict_fun = {1: lambda x,w : np.vstack((w[1],w[2],w[3], (-self.q(x) - self.k(x) * w[0] - self.N * w[2] ) / ( self.E * self.Jx))),
    2: lambda x,w : np.vstack((w[1],w[2],w[3], (-self.q(x)*np.cos(self.phi)- self.k(x) * w[0] - (self.N+self.q(x)*np.sin(self.phi)) * w[2] ) / ( self.E * self.Jx)))}
    return  dict_fun[self.task_id]
  def bc_maker(self):
    dict_bc = {0 : lambda ya,yb : np.array([ya[0],yb[0]]),
    1: lambda ya,yb : np.array([ya[0], yb[0], ya[2], yb[2]]),
    2: lambda ya,yb : np.array([ya[0], yb[0], ya[1], yb[1]]),
    3: lambda ya,yb : np.array([ya[0],yb[0],ya[1],yb[2]]),
    4: lambda ya,yb : np.array([ya[0],yb[0],ya[2],yb[1]]),
    5: lambda ya,yb : np.array([ya[0],yb[0],ya[3],yb[3]])}
    return dict_bc[self.bc_id]
  @property
  def task_id(self):
    return self.__task_id
  @task_id.setter
  def task_id(self,task_id):
    if task_id in self.dict_id_dimensions.keys():
          self.__task_id = task_id
    else:
          raise ValueError('Wrong task_id')
  @property
  def bc_id(self):
    return self.__bc_id
  @bc_id.setter
  def bc_id(self,id):
    if id in self.bc_id_dimensions.keys() and self.bc_id_dimensions[id] == self.dict_id_dimensions[self.task_id]:
      self.__bc_id = id
    else:
      print('bc_id and task_id doesnt match. Set it to default for this task_id')
      self.__bc_id = 1 if self.dict_id_dimensions[self.task_id] == 4 else 0
  @property
  def totalNodes(self):
    return self.__totalNodes
  @totalNodes.setter
  def totalNodes(self,totalNodes):
    if totalNodes!=0:
      self.__totalNodes = abs(totalNodes)
    else:
      raise ValueError("totalNodes can't be 0")
  @property
  def x(self):
    return np.linspace(0, self.l, self.totalNodes)
  @property
  def y(self):
    return np.zeros((self.dict_id_dimensions[self.task_id], self.x.shape[0]))
  @property
  def solution(self):
    return solve_bvp(self.fun_maker(),self.bc_maker(),self.x,self.y, tol = 1e-10, max_nodes=self.totalNodes)
  def q(self,x):
    temp1,temp2 = 0,0
    dim = len(x)
    #print(dim)
    if len(self.addQGauss_params) > 0:
      addQGauss_array = []
      for i in range(len(self.addQGauss_params)):
        addQGauss_array.append(
            self.addQGauss_params[i]['P'] / math.sqrt(2 * math.pi) / self.addQGauss_params[i]['sigma']* np.exp(- (x - self.addQGauss_params[i]['x0']) ** 2 /
                                                                                                         2 / self.addQGauss_params[i]['sigma'] ** 2) )
      temp1 +=sum(addQGauss_array)
    if len(self.addQ_params) > 0:
      addQ_array = []
      for i in range(len(self.addQ_params)):
        array = []
        for j in range(dim):
          array.append(self.addQ_params[i]['P'] if abs(x[j]-self.addQ_params[i]['x0'])<=self.addQ_params[i]['sigma'] else 0)
        addQ_array.append(array)
      addQ_array = np.array(addQ_array)
      temp2 += sum(addQ_array)
    Q = self.m * 9.8 / self.l if self.Q_on else 0
    return Q + temp1 + temp2 #погонный вес трубы

  def k(self,x):
    return 40 * 10**6 * 2 * np.sqrt(self.q(x) * (1 - 0.09) * self.R / np.pi / self.E_soil)
  @property
  def addQ_params(self):
    return self.__addQ_params
  @addQ_params.setter
  def addQ_params(self,params):
    if type(params) is not list:
      self.__addQ_params = [{'x0': 0, 'sigma' : 0, 'P' : 0}]
      print('addQ_params didnt set.Made it zero')
    else:
      self.__addQ_params = params
  @property
  def addQGauss_params(self):
    return self.__addQGauss_params
  @addQGauss_params.setter
  def addQGauss_params(self,params):
    if type(params) is not list:
      self.__addQGauss_params = [{'x0': 0, 'sigma' : 0, 'P' : 0}]
      print('addQGauss_params didnt set.Made it zero')
    else:
      self.__addQGauss_params = params
  @property
  def Q_on(self):
    return self.__Q_on
  @Q_on.setter
  def Q_on(self,On):
    if type(On) is not bool:
      self.__Q_on = True
      print('Q_on didnt set correct. Made it True')
    else:
      self.__Q_on = On
  def max_deflection(self):
    y_measles_plot = self.solution.sol(self.solution.x)
    ans = round(max(abs(y_measles_plot[0])),6)
    return ans