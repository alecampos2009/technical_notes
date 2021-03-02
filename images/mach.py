import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')

params = {'font.size' : 25.0,
          'legend.fontsize': 'small',
          'axes.labelsize': 'large',
          'axes.titlesize': 'large',
          'xtick.labelsize':'small',
          'ytick.labelsize':'small',
          'lines.linewidth': 4,
          'figure.figsize': (8, 6),}

plt.rcParams.update(params)

Mach = np.array([1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 5.0])
gam = 1.4

n_elem = 100
beta = np.zeros([Mach.size + 1, n_elem])
theta = np.zeros([Mach.size + 1, n_elem])

for i in range(Mach.size):
	mu = np.arcsin(1.0 / Mach[i])
	beta[i,:] = np.linspace(mu, 0.5 * np.pi, n_elem)

	for j in range(n_elem):
		num = 1.0 / (np.tan(beta[i,j])) * (Mach[i]**2 * np.sin(beta[i,j])**2 - 1.0)
		den = 1.0 + 0.5 * (gam + 1.0) * Mach[i]**2 - Mach[i]**2 * np.sin(beta[i,j])**2
		theta[i,j] = np.arctan(num / den)

beta[-1,:] = np.linspace(0.0, 0.5 * np.pi, n_elem)
for j in range(n_elem):
	num = np.cos(beta[-1,j]) * np.sin(beta[-1,j])
	den = 0.5 * (gam + 1.0) - np.sin(beta[-1,j])**2
	theta[-1,j] = np.arctan(num / den)

boundary_x = np.zeros(Mach.size + 1)
boundary_y = np.zeros(Mach.size + 1)
for i in range(Mach.size + 1):
	boundary_y[i] = np.max(theta[i,:])
	boundary_x[i] = beta[i,np.argmax(theta[i,:])]

plt.figure()
plt.plot(beta[2,:], theta[2,:], label=r'$M = 1.5$')
plt.plot(beta[5,:], theta[5,:], label=r'$M = 2$')
plt.plot(beta[7,:], theta[7,:], label=r'$M = 3$')
plt.plot(beta[9,:], theta[9,:], label=r'$M = 5$')
plt.plot(beta[-1,:], theta[-1,:], label=r'$M = \infty$')
plt.plot(boundary_x, boundary_y, '--', color='k')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\theta$')
plt.xlim([0, 0.5 * np.pi])
plt.ylim([0, 0.9])
plt.tight_layout()
plt.tick_params(direction='in', top=True, right=True)
plt.text(0.7, 0.3,'weak \n solution', horizontalalignment='center', bbox=dict(facecolor='white'))
plt.text(1.4, 0.55, 'strong \n solution', horizontalalignment='center', bbox=dict(facecolor='white'))
plt.text(1.05, 0.52, r'$\theta_{max}$ line', rotation=90)
plt.text(1.50, -0.1, r'$\pi / 2$')
plt.text(1.59, 0.01, r'a')
plt.text(1.17, 0.22, r'b')
plt.text(1.05, 0.21, r'c')
plt.text(0.67, 0.01, r'd')
plt.legend(frameon=False)

plt.show()