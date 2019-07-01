import functions as aux
from PyDSTool import *
import PyDSTool as dst
import numpy as np
import copy as cp






bif_startpoint=0.1
maxstep=1e+4
minstep=1e-1
step=5e+2
maxpoints=500
LocBifPoints=['LP','B']
freepar = 'It'



# Bifurcation diagram for control (i.e. no NRF2)

DSargs          = args(name='Notch_EMT_1cell', checklevel=2)
DSargs.pars     = aux.parameters(onecell=True, full_model=False)
DSargs.varspecs = aux.equations(onecell=True, full_model=False)
DSargs.fnspecs  = aux.functions()

DSargs.ics = {'W': 20000.0, 'Z': 40000.0,'Y': 20000.0,'S': 200000.0,'N':  0.0e+0, 'D': 0.0e+0, 'J': 0.0e+0, 'I': 20.0e+0}
DSargs.xdomain  = {'W': [0, 5.0e+4],'Z': [0, 5.0e+6],'Y': [0, 5.0e+4],'S': [0, 5.0e+5],'N': [0, 5.0e+5],'D': [0, 5.0e+5],'J': [0, 5.0e+5],'I': [0, 5.0e+5]}

DSargs.pdomain  = {'Dt': [-0.1e+0, 1.0e+3],'Jt': [-0.1e+0, 1.1e+4],'It': [-0.1e+0, 5.1e+3]}
DSargs.tdomain  = [0., 800.0]
DSargs.algparams= {'init_step':1.0e-1}



ODE = Vode_ODEsystem(DSargs)
ODE.set(pars = {'Nt': 0., 'gD': 40, 'gJ': 15, 'It': 0.1, 'Dt': 0.1, 'Jt': 0.})
ODE.set(pars={'b_J': 1, 'b_D': 1, 'b_N': 1, 'lNICD':1.0})

fp=aux.fast_fixedpoint(ODE)
ics=[fp]
ODE.set(pars = {freepar: bif_startpoint})
PCargs = aux.PyCont_args(ODE.name, freepar, maxpoints+100, saveeigen=True, LocBifPoints=LocBifPoints,maxstep=maxstep, minstep=minstep, step=step)
for j in range(len(ics)):
	ODE.set(ics  = ics[j])
	PyCont = PyDSTool.ContClass(ODE)
	PyCont.newCurve(PCargs)
	PyCont[ODE.name].forward()
	PyCont[ODE.name].backward()
	PyCont.display((freepar,'W'), stability=True, axes=ax ,color='k', linewidth=4)
	PyCont.plot.toggleLabels('off')



# Bifurcation diagram for NRF2 case

DSargs          = args(name='Notch_EMT_1cell', checklevel=2)
DSargs.pars     = aux.parameters(onecell=True, full_model=True)
DSargs.varspecs = aux.equations(onecell=True, full_model=True)
DSargs.fnspecs  = aux.functions()

DSargs.ics = {'W': 38000.0, 'Z': 40000.0,'Y': 20000.0,'S': 200000.0,'N':  0.0e+0, 'D': 0.0e+0, 'J': 0.0e+0, 'I': 20.0e+0, 'nrf': 500000., 'Ec':250000., 'keap':250000.}
	
DSargs.xdomain  = {'W': [0, 5.0e+4],'Z': [0, 5.0e+6],'Y': [0, 5.0e+4],'S': [0, 5.0e+5],'N': [0, 5.0e+5],'D': [0, 5.0e+5],'J': [0, 5.0e+5],'I': [0, 5.0e+5], 'nrf':[0, 10.0e+6], 'Ec':[0.,1.0e+6], 'keap':[0.,1.0e+6]}

DSargs.pdomain  = {'Dt': [-0.1e+0, 1.0e+3],'Jt': [-0.1e+0, 1.1e+4],'It': [-0.1e+0, 5.1e+3]}
DSargs.tdomain  = [0., 100.0]
DSargs.algparams= {'init_step':1.0e-1}


ODE = Vode_ODEsystem(DSargs)
ODE.set(pars = {'Nt': 0., 'gD': 40, 'gJ': 15, 'It': 0.1, 'Dt': 0.1, 'Jt': 0.})
ODE.set(pars={'b_J': 1, 'b_D': 1, 'b_N': 1, 'lNICD':1.0})
ODE.set(pars={'l_nrf_I':1.0, 'l_N_nrf':1.0})

ODE.set(pars={'l_S_N':0.5, 'nrf_S':3.0e6})

fp=aux.fast_fixedpoint(ODE)
ics=[fp]
ODE.set(pars = {freepar: bif_startpoint})
PCargs = aux.PyCont_args(ODE.name, freepar, maxpoints, saveeigen=True, LocBifPoints=LocBifPoints,maxstep=maxstep, minstep=minstep, step=step)
for j in range(len(ics)):
	ODE.set(ics  = ics[j])
	PyCont = PyDSTool.ContClass(ODE)
	PyCont.newCurve(PCargs)
	PyCont[ODE.name].forward()
	PyCont[ODE.name].backward()
	PyCont.display((freepar,'W'), stability=True, axes=ax ,color='r', linewidth=4)
	PyCont.display((freepar,'nrf'), stability=True, axes=ax2 ,color='r', linewidth=4)
	PyCont.plot.toggleLabels('off')




















