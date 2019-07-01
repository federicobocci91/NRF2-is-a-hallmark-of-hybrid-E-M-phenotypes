from PyDSTool import *
from PyDSTool.Toolbox import phaseplane as pp
from matplotlib import cm
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import random as rand
import copy as cp
from scipy.ndimage import measurements
import matplotlib.colors as cl
import math
from matplotlib.font_manager import FontProperties


###########################################################################################################################
#																														  #
# functions adapted from Marcelo Boareto et al, JRSI 2016 https://github.com/mboareto/Notch_EMT_JRSocInterface2016        #
#																														  #
###########################################################################################################################


def parameters(onecell=False, full_model = False):
	
	dic =  {'km' : 5.0e-1,'kP' : 1.0e+2,'gu200' : 2.1e+3,'gZ': 1.1e+1,'Z0u200': 2.2e+5,'Z0Z': 2.5e+4,'nZu200': 3.0e+0,'nZZ': 2.0e+0, 'nu200' : 6.0e+0, 'nSu200': 2.0e+0, 'nSZ': 2.0e+0,'lZu200': 1.0e-1, 'lZZ': 7.5e+0, 'lSu200': 1.0e-1, 'lSZ'   : 1.0e+1,'ku200' : 5.0e-2, 'kZ' : 1.0e-1,'S0u200': 1.8e+5, 'S0Z': 1.8e+5,'u0200' : 1.0e+4,'nSu34' : 1.0e+0, 'nSS': 1.0e+0, 'nu34' : 2.0e+0, 'nI' : 2.0e+0,'lSu34' : 1.0e-1, 'lSS': 1.0e-1, 'lZu34': 2.0e-1, 'lIS': 6.5e+0, 'lNICD' : 6.5e+0 ,'ku34'  : 5.0e-2, 'kS' : 1.25e-1 ,'gu34'  : 1.35e+3,'gS' : 9.0e+1 ,'S0u34' : 3.0e+5, 'S0S': 2.0e+5 ,'Z0u34' : 6.0e+5 ,'u034'  : 1.0e+4,'I0S'   : 3.0e+2
		   #----------Notch Signaling circuit----------
		,'k'  : 1.0e-1, 'kI' : 5.0e-1,'kc' : 1.0e-4, 'kt' : 1.0e-5,'p'  : 2.0e+0, 'pf' : 1.0 ,'gN' : 0.8e+1, 'gD' : 7.0e+1, 'gJ' : 2.0e+1 ,'I0' : 1.0e+2 ,'Nt' : 0.0e+0, 'Dt' : 0.0e+0, 'Jt' : 0.0e+0 ,'Nn' : 0.0e+0, 'Dn' : 0.0e+0, 'Jn' : 0.0e+0 ,'ln' : 7.0e+0, 'ld' : 0.0e+0, 'lj' : 2.0e+0 ,'ldf': 3.0,    'ljf': 0.3,'It' : 0.0}

	if onecell:
		dic.update({'l0': 1.0e+0,'l1': 6.0e-1,'l2': 3.0e-1,'l3': 1.0e-1,'l4': 5.0e-2,'l5': 5.0e-2,'l6': 5.0e-2,'gm0': 0.0e+0,'gm1': 4.0e-2,'gm2': 2.0e-1,'gm3': 1.0e+0,'gm4': 1.0e+0,'gm5': 1.0e+0,'gm6': 1.0e+0,'gu0': 0.0e+0,'gu1': 1*5.0e-3,'gu2': 2*5.0e-2,'gu3': 3*5.0e-1,'gu4': 4*5.0e-1,'gu5': 5*5.0e-1,'gu6': 6*5.0e-1})
	else:
		dic.update({'l' : [1.0e+0, 6.0e-1, 3.0e-1, 1.0e-1, 5.0e-2, 5.0e-2, 5.0e-2],'gm': [0.0e+0, 4.0e-2, 2.0e-1, 1.0e+0, 1.0e+0, 1.0e+0, 1.0e+0],'gu': [0.0e+0, 1*5.0e-3, 2*5.0e-2, 3*5.0e-1, 4*5.0e-1, 5*5.0e-1, 6*5.0e-1],'HS' : HS, 'Pl' : Pl, 'Py' : Py})

	dic.update({'b_J': 5, 'b_D': 3, 'b_N': 2, 'b_Z': 6, 'b_S': 2, })


	if full_model:
		
		dic.update({'g_nrf':0.5e5, 'k_nrf':0.1}) # nrf2 production and degradation
		# notch-nrf2 axis
		dic.update({'n_nrf_I':2, 'l_nrf_I':2.5}) # Nicd on Nrf2
		dic.update({'nrf_0':0.75e6, 'n_N_nrf':2, 'l_N_nrf':2.5}) # Nrf2 on Notch
		#nrf2-to-EMT
		dic.update({'nrf_S':1.0e6 ,'n_S_N': 2, 'l_S_N':0.67}) # nrf2 on snail
		# EMT-to-nrf2
		dic.update({'gE':50000, 'kE':0.1, 'n_E_Z':2, 'l_Z_E':0.1, 'Z0_E':20000}) # E-cad equation
		dic.update({'x1':250000, 'x2':2, 'x3':0.33})
		
		dic.update({'gk':50000, 'kk':0.1, 'n_W_k':2, 'l_W_k':0.1, 'W0_k':5000}) # keap equation
		dic.update({'k0_nrf':250000, 'n_k_nrf':2, 'l_k_nrf':0.33}) # keap on nrf2

	return dic


def equations(onecell=False, full_model = False):
	
	if full_model:
		return {
			'W' : 'gu200*HS(Z,Z0u200,nZu200,lZu200)*HS(S,S0u200,nSu200,lSu200) - gZ*HS(Z,Z0Z,nZZ,lZZ)*HS(S,S0Z,nSZ,lSZ)*Py(W,km,b_Z) - gJ*HS(I,I0,p,lj)*Py(W,km,b_J) - ku200*W',
			'Y' : 'gu34*HS(S,S0u34,nSu34,lSu34)*HS(Z,Z0u34,nu34,lZu34) - gS*HS(S,S0S,nSS,lSS)*HS(I,I0S,nI,lIS)*HS(It,I0S,nI,lIS)*Py(Y,km,b_S) - gN*HS(I,I0,p,ln)*Py(Y,km,b_N) - gD*HS(I,I0,p,ld)*Py(Y,km,b_D) - ku34*Y',
			'Z' : 'kP*gZ*HS(Z,Z0Z,nZZ,lZZ)*HS(S,S0Z,nSZ,lSZ)*Pl(W,km,b_Z) - kZ*Z',
			'S' : 'kP*gS*HS(S,S0S,nSS,lSS)*HS(I,I0S,nI ,lNICD)*HS(It,I0S,nI,lIS)*Pl(Y,km,b_S)*HS(nrf, nrf_S, n_S_N, l_S_N)  - kS*S',
			'N' : 'kP*gN*HS(I,I0,p,ln)*Pl(Y,km,b_N)*HS(nrf,nrf_0,n_N_nrf,l_N_nrf) - N*( (kc*D + kt*Dt)*HS(I,I0,pf,ldf) + (kc*J + kt*Jt)*HS(I,I0,pf,ljf) ) - k*N',
			'D' : 'kP*gD*HS(I,I0,p,ld)*Pl(Y,km,b_D) - D*(  kc*N*HS(I,I0,pf,ldf) + kt*Nt ) - k*D',
			'J' : 'kP*gJ*HS(I,I0,p,lj)*Pl(W,km,b_J) - J*(  kc*N*HS(I,I0,pf,ljf) + kt*Nt ) - k*J',
			'I' : 'kt*N*( Dt*HS(I,I0,pf,ldf) + Jt*HS(I,I0,pf,ljf) ) - kI*I',
			'Ec': 'gE*HS(Z,Z0_E,n_E_Z, l_Z_E) - kE*Ec',
			'nrf': 'g_nrf*HS(I,I0,n_nrf_I,l_nrf_I)*HS(keap,k0_nrf,n_k_nrf,l_k_nrf)*HS(Ec,x1,x2,x3) - k_nrf*nrf ',
			'keap':'gk*HS(W,W0_k,n_W_k, l_W_k) - kk*keap'}

	else:
		return {
			'W' : 'gu200*HS(Z,Z0u200,nZu200,lZu200)*HS(S,S0u200,nSu200,lSu200) - gZ*HS(Z,Z0Z,nZZ,lZZ)*HS(S,S0Z,nSZ,lSZ)*Py(W,km,b_Z) - gJ*HS(I,I0,p,lj)*Py(W,km,b_J) - ku200*W',
			'Y' : 'gu34*HS(S,S0u34,nSu34,lSu34)*HS(Z,Z0u34,nu34,lZu34) - gS*HS(S,S0S,nSS,lSS)*HS(I,I0S,nI,lIS)*HS(It,I0S,nI,lIS)*Py(Y,km,b_S) - gN*HS(I,I0,p,ln)*Py(Y,km,b_N) - gD*HS(I,I0,p,ld)*Py(Y,km,b_D) - ku34*Y',
			'Z' : 'kP*gZ*HS(Z,Z0Z,nZZ,lZZ)*HS(S,S0Z,nSZ,lSZ)*Pl(W,km,b_Z) - kZ*Z',
			'S' : 'kP*gS*HS(S,S0S,nSS,lSS)*HS(I,I0S,nI ,lNICD)*HS(It,I0S,nI,lIS)*Pl(Y,km,b_S)  - kS*S',
			'N' : 'kP*gN*HS(I,I0,p,ln)*Pl(Y,km,b_N) - N*( (kc*D + kt*Dt)*HS(I,I0,pf,ldf) + (kc*J + kt*Jt)*HS(I,I0,pf,ljf) ) - k*N',
			'D' : 'kP*gD*HS(I,I0,p,ld)*Pl(Y,km,b_D) - D*(  kc*N*HS(I,I0,pf,ldf) + kt*Nt ) - k*D',
			'J' : 'kP*gJ*HS(I,I0,p,lj)*Pl(W,km,b_J) - J*(  kc*N*HS(I,I0,pf,ljf) + kt*Nt ) - k*J',
			'I' : 'kt*N*( Dt*HS(I,I0,pf,ldf) + Jt*HS(I,I0,pf,ljf) ) - kI*I'}




def HS(X,X0,nX,lamb):
	return lamb + (1.0-lamb)/(1.0 + (X/X0)**nX)

def M(X,X0,i,n):
	return ((X/X0)**i)/((1. + (X/X0))**n)

def C(i,n):
	return gamma(n+1)/(gamma(n-i+1)*gamma(i+1))


def Py(X, n, k, u0, gu=[0.0e+0, 1*5.0e-3, 2*5.0e-2, 3*5.0e-1, 4*5.0e-1, 5*5.0e-1, 6*5.0e-1],gm=[0.0e+0, 4.0e-2, 2.0e-1, 1.0e+0, 1.0e+0, 1.0e+0, 1.0e+0]):
	v1 = 0
	v2 = 0
	for i in range(n+1):
		v1 += gu[i]*C(i,n)*M(X,u0,i,n)
		v2 += gm[i]*C(i,n)*M(X,u0,i,n)
	return v1/(v2+k)

def Pl(X, n, k, u0, l=[1.0e+0, 6.0e-1, 3.0e-1, 1.0e-1, 5.0e-2, 5.0e-2, 5.0e-2],gm=[0.0e+0, 4.0e-2, 2.0e-1, 1.0e+0, 1.0e+0, 1.0e+0, 1.0e+0]):
	v1 = 0
	v2 = 0
	for i in range(n+1):
		v1 +=  l[i]*C(i,n)*M(X,u0,i,n)
		v2 += gm[i]*C(i,n)*M(X,u0,i,n)
	return v1/(v2+k)


def functions():
	return {'HS': (['X','X0','nX','lamb'], 'lamb + (1.0-lamb)/(1.0 + (X/X0)**nX)'),'M' : (['X','X0','i','n'], '((X/X0)**i)/((1. + (X/X0))**n)' ),'C' : (['i','n'],'special_gamma(n+1)/(special_gamma(n-i+1)*special_gamma(i+1))'     ),'Py': (['X','kd','n'],'sum(i, 0, 6, if([i]>n, 0, gu[i]*C([i],n)*M(X,u0200,[i],n)))/(sum(i, 0, 6, if([i]>n, 0, gm[i]*C([i],n)*M(X,u0200,[i],n))) + kd)'),'Pl': (['X','kd','n'],'sum(i, 0, 6, if([i]>n, 0,  l[i]*C([i],n)*M(X,u0200,[i],n)))/(sum(i, 0, 6, if([i]>n, 0, gm[i]*C([i],n)*M(X,u0200,[i],n))) + kd)')}


def euler_traj(eqs, p, pts=None, vlim=None, hexagonal=True,
			   nsignal_dict={'N': ['D', 'J'], 'I': ['D', 'J'], 'D': ['N'], 'J': ['N']}):
	if pts==None:
		if vlim==None:
			print 'ERROR: Give me a starting point (pts) or the limits for a random start point (vlim)'
			return 0
		pts = {}
		for j in eqs.keys():
			pts[j] = np.random.uniform(vlim[j][0],vlim[j][1],(p['n'],p['n']))
		
	pts_new = {}
	for t in range(int(p['t']/p['dt'])):
		for key in eqs.keys():
			if key in nsignal_dict.keys():
				for k in nsignal_dict[key]:
					p[k+'n'] = nsignal_sum(p, pts, k, key, hexagonal=hexagonal)
			pts_new[key] = pts[key] + p['dt']*eval(eqs[key], p, pts)
		pts = pts_new
	return pts


def eliminate_redundants(fp, eps=10):
	for i in range(len(fp)):
		for k, v in fp[i].items():
			v = round(v,eps)
			fp[i][k] = v
	seen = set()
	new_l = []
	for d in fp:
		t = tuple(d.items())
		if t not in seen:
			seen.add(t)
			new_l.append(d)
	return new_l

def stability(FPs, ODE, eps=0.1):
	out = []
	for i in range(len(FPs)):
		X = {}
		stable = True
		for k in FPs[0].keys():
			X[k] = FPs[i][k]*(1 + eps*rand.sample(list([-1,1]),1)[0])
		ODE.set(ics  = X)
		traj = ODE.compute('traj')
		X = traj.sample()[-1]
		for k in FPs[0].keys():
			if np.abs(X[k]-FPs[i][k]) > eps*FPs[i][k]:
				stable = False
		out += ['S'] if stable else ['I']
	return out

def PyCont_args(nmodel, freepar, maxnumpoints, maxstep=1e+1, minstep=1e-1, stopAt=['B'],step=1e-0, LocBifPoints=['BP','LP','B'], saveeigen=False, Type='EP-C'):
	PCargs = PyDSTool.args(name=nmodel, type=Type)    # 'EP-C' stands for Equilibrium Point Curve.
	PCargs.freepars     = [freepar]                   # control parameter
	PCargs.MaxNumPoints = maxnumpoints                # The following 3 parameters are set after trial-and-error
	PCargs.MaxStepSize  = maxstep
	PCargs.MinStepSize  = minstep
	PCargs.StepSize     = step
	PCargs.StopAtPoints = stopAt
	PCargs.LocBifPoints = LocBifPoints                # detect limit points / saddle-node bifurcations
	PCargs.SaveEigen    = saveeigen                   # to tell unstable from stable branches
	return PCargs

def fast_fixedpoint(ODE, tdomain=[0, 100000]):
	ODE.set(tdomain=tdomain)
	traj = ODE.compute('traj')
	pts = traj.sample()
	return dict(pts[-1])








