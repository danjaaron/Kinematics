import numpy as np
import matplotlib.pyplot as plt
import os, pickle
# f = lambda x: float(x**2 + 10)
# df = lambda x: float(2*x)
# df_normalized = lambda x: float(np.sign(x))
# x = 100

x = np.array([10, 1, 2])
A = np.array([[1, 0, 0], [0, 20, 0], [0, 0, .01]])
f = lambda x: x.T.dot(A).dot(x) + 100
df = lambda x: 2 * A.dot(x)
df_normalized = lambda x: df(x) / np.linalg.norm(df(x))

iters = 100

def norm(data):
	return [_/(max(data)) for _ in data]

def normjoint(datalist):
	# normalize lists against each other
	maxlist = [max(k) for k in datalist]
	jointmax = max(maxlist)
	newdatalist = [k/jointmax for k in datalist]
	return newdatalist

def optimize(x, f, df, df_normalized, T=iters, thresh=1e-3, verbose = False): #1e-3, verbose = False):
	g = 10.0
	larr, ssarr, garr = [], [], []
	dx = - np.sqrt(2.0 * float(f(x))/g) * df_normalized(x)
	for i in range(T):
		f_orig = f(x)
		step_size = np.sqrt(2.0 * float(f(x))/g)
		larr.append(f_orig)
		garr.append(g)
		ssarr.append(step_size)
		x = x + - np.sqrt(2.0 * float(f(x))/g) * df_normalized(x)
		if verbose:
			print(i, x, f(x), step_size, g)
		if f(x) >= f_orig - thresh:
			# stability probs at g = 1
			g *= 2
	return larr, garr

def weird_mnm_optimize(x, f, df, df_normalized, T=iters, thresh=1e-3, verbose = False):
	g = 10.0

	step_size = np.sqrt(2.0 * float(f(x))/g)
	v = 0 #-step_size*df_normalized(x)

	# double_add = True
	larr, ssarr, garr = [], [], []
	for i in range(T):
		f_orig = f(x)
		step_size = np.sqrt(2.0 * float(f(x))/g)
		larr.append(f_orig)
		garr.append(g)
		ssarr.append(step_size)
		if i == 0:
			v = -step_size*df_normalized(x)
		x = x + v 
		v = -step_size*df_normalized(x)
		
		# if double_add:
			# x = x - step_size * df_normalized(x)
		if verbose:
			print(i, x, f(x), step_size, g)
		if f(x) >= f_orig:
			g *= 2
	return larr, garr

def mnm_optimize(x, f, df, df_normalized, T=iters, thresh=1e-3, verbose = False):
	g = 10.0
	beta = 0.9

	step_size = np.sqrt(2.0 * float(f(x))/g)
	v = 0. #-step_size*df_normalized(x)
	
	larr, ssarr, garr = [], [], []
	for i in range(T):
		f_orig = f(x)
		step_size = np.sqrt(2.0 * float(f(x))/g)
		larr.append(f_orig)
		garr.append(g)
		ssarr.append(step_size)
		# if i == 0:
			# v = -step_size*df_normalized(x)

		v = (1-beta)*v + beta*(-step_size*df_normalized(x))

		x = x + v #- step_size * df_normalized(x)
		# if beta == 0.:
			# assert(all([v == -step_size*df_normalized(x)[i] for (i, v) in enumerate(v)]))
		# if double_add:
			# x = x - step_size * df_normalized(x)
		# if i > 0:
			# v = -step_size*df_normalized(x) + beta*v
		if verbose:
			print(i, x, f(x), step_size, g)
		if f(x) >= f_orig:
			g *= 2
		# elif f(x) < f_orig:
			# g *= 0.5
	return larr, garr

def p_optimize(x0, f, df, df_normalized, T=iters, verbose = False):
	# increase g if less progress is made than last step
	g = 1e-10
	scale_factor = 10.0
	larr, ssarr, garr = [], [], []
	progress = [0.]
	for i in range(T):
		
		l0 = f(x0)
		step_size = np.sqrt(2.0 * float(l0)/g)

		
		
		larr.append(l0)
		garr.append(g)
		ssarr.append(step_size)

		print(df_normalized(x0))
		print(df(x0))

		xf = x0 - step_size * df_normalized(x0)

		lf = f(xf) 

		prev_p = progress[i] 
		curr_p = float(l0 - lf) # progress in this step

		while ((curr_p < 0.0)): # and (g <= abs(lf/l0))): 
			g *= scale_factor
			step_size = np.sqrt(2.0 * float(l0)/g)
			xf = x0 - step_size * df_normalized(x0)
			lf = f(xf) 
			curr_p = float(l0 - lf)
			if verbose:
				print("repeated ", lf, curr_p, prev_p, g)
		curr_p = float(l0 - lf) # progress in this step
		progress.append(curr_p)

		# # increase g if last step made more progress
		# if (prev_p < curr_p):
		# 	g *= 10.
		# else:
		# 	g *= 0.1
		
		if lf >= l0 - lf/float(g):
			if verbose:
				print('g +')
			g *= scale_factor

		if verbose:
			print(i, lf, x0, xf, curr_p, prev_p, g)
		x0 = xf

	return larr, garr

def pw_optimize(x0, f, df, df_normalized, T=iters, verbose = False):
	# increase g if less progress is made than last step
	g = 1e-5
	larr, ssarr, garr = [], [], []
	progress = [0.]
	for i in range(T):
		
		l0 = f(x0)
		step_size = np.sqrt(2.0 * float(l0)/g)

		
		
		larr.append(l0)
		garr.append(g)
		ssarr.append(step_size)

		xf = x0 - step_size * df_normalized(x0)

		lf = f(xf) 

		prev_p = progress[i] 
		curr_p = float(l0 - lf) # progress in this step

		while ((curr_p < 0.0)): # and (g <= abs(lf/l0))): 
			g *= (10.)
			step_size = np.sqrt(2.0 * float(l0)/g)
			xf = x0 - step_size * df_normalized(x0)
			lf = f(xf) 
			curr_p = float(l0 - lf)
			if verbose:
				print("repeated ", lf, curr_p, prev_p, g)
		curr_p = float(l0 - lf) # progress in this step
		progress.append(curr_p)

		# increase g if last step made more progress
		if (prev_p < curr_p):
			g *= 10.
		else:
			g *= 0.1
		# if lf >= l0 - lf/float(g):
		# 	if verbose:
		# 		print('g +')
		# 	g *= 2.0
		if verbose:
			print(i, lf, x0, xf, curr_p, prev_p, g)
		x0 = xf

	return larr, garr

def sgd(x, f, df, df_normalized, T=iters, step_size=1e-3, verbose = False):
	larr = []
	for i in range(T):
		f_orig = f(x)
		larr.append(f_orig)
		x = x - step_size * df(x)
		if verbose:
			print(i, x)
	return larr

normalize_bool = False
plot_g = False

# p_optimize(x, f, df, df_normalized, verbose = True)


def run_optimizers(opt_list):
	results = [o(x, f, df, df_normalized) for o in opt_list]
	loss_arr = [r[0] for r in results]
	g_arr = [g[1] for g in results]
	return [loss_arr, g_arr]

def plot_optimizers(opt_list, colors, legend_list):
	[l, g] = run_optimizers(opt_list)
	for (li, l_) in enumerate(l):
		plt.plot(l_, c = colors[li])
		print("graph {}: max g is {}".format(li, max(g[li])))
	# print(g[1][0])
	plt.legend(legend_list)
	plt.title("Optimizer Loss vs Iters")
	plt.show()

opts = [optimize, mnm_optimize]
colors = ['black', 'green', 'red']
lgnd = ['threshold', 'momentum']
plot_optimizers(opts, colors, lgnd)

'''
# normalizations
if normalize_bool:
	tg_plot = norm(tg_l)
	g_plot = norm(g_l)
	pg_plot = norm(pg_l)
	pkin_plot, tkin_plot, kin_plot, sgd_plot = normjoint([pkin_l, tkin_l, kin_l, sgd_l])
else:
	tg_plot = (tg_l)
	g_plot = (g_l)
	pg_plot = (pg_l)
	pkin_plot, tkin_plot, kin_plot, sgd_plot = [pkin_l, tkin_l, kin_l, sgd_l]

# plot kin losses
plt.plot(kin_plot, 'blue')
plt.plot(tkin_plot, 'red')
plt.plot(pkin_plot, 'green')

# plot g
if plot_g:
	plt.plot(g_plot, 'b--')
	plt.plot(tg_plot, 'r--')
	plt.plot(pg_plot, 'g--')

# plot sgd for comparison
plt.plot(sgd_plot, 'black')

if plot_g:
	plt.legend(['k', 't', 'p', 'g', 'tg', 'pg', 'sgd'])
else:
	plt.legend(['k', 't', 'p', 'sgd'])	

plt.show()
'''
# print(kin_l)

# plt.plot(list(range(len(kin_l))), kin_l)
