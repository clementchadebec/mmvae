import seaborn as sb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame( 
                  dict(
                      n_made = [0,1,2,3,5], 
                      acc01 = [0.79,0.8285,0.83 ,0.8341,0.8317], 
                      acc10 = [0.46,0.5422, 0.55,0.58,0.5791], 
                      std10= [0.001,0.0024, 0.0012,0.0012,0.0012],
                      std01=[0.001,0.001, 0.001,0.001,0.002],
                      )
                  )


print(df)


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Number of MADE blocks')
ax1.set_ylabel(r'$P_S$', color=color)
ax1.errorbar(df['n_made'], df['acc01'], yerr=df['std01'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel(r'$P_M$', color=color)  # we already handled the x-label with ax1
ax2.errorbar(df['n_made'], df['acc10'], yerr=df['std10'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.title('Influence of the number of MADE blocks')
plt.savefig('n_made_jnf_ms.pdf')

df = pd.DataFrame( 
                  dict(
                      n_made = [0,1,2,3,5], 
                      fid0 = [13,11.271,10.6 ,10.46,10.62], 
                      fid1 = [72,70.0, 66.71,70.2,69.7], 
                      std0= [0.05,0.0024, 0.08,0.01,0.05],
                      std1=[0.05,0.03, 0.03,0.001,0.002],
                      )
                  )


print(df)


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Number of MADE blocks')
ax1.set_ylabel(r'$FID_M$', color=color)
ax1.errorbar(df['n_made'], df['fid0'], yerr=df['std0'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel(r'$FID_S$', color=color)  # we already handled the x-label with ax1
ax2.errorbar(df['n_made'], df['fid1'], yerr=df['std1'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.title('Influence of the number of MADE blocks')
plt.savefig('n_made_jnf_ms_fid.pdf')



df = pd.DataFrame( 
                  dict(
                      n_made = [1,2,3,5], 
                      acc01 = [0.7906,0.8015,0.8041 ,0.8086], 
                      acc10 = [0.7585,0.7849, 0.7922,0.7993], 
                      std10= [0.001,0.001, 0.001,0.001],
                      std01=[0.002,0.001, 0.001,0.001],
                      )
                  )


print(df)


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Number of MADE blocks')
ax1.set_ylabel(r'$P_S$', color=color)
ax1.errorbar(df['n_made'], df['acc01'], yerr=df['std01'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel(r'$P_M$', color=color)  # we already handled the x-label with ax1
ax2.errorbar(df['n_made'], df['acc10'], yerr=df['std10'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.title('Influence of the number of MADE blocks')
plt.savefig('n_made_jnfd_ms.pdf')

df = pd.DataFrame( 
                  dict(
                      n_made = [1,2,3,5], 
                      fid0 = [10.544,10.392,10.877 ,10.51], 
                      fid1 = [70.41,69.754, 71.107,69.593], 
                      std0= [0.03,0.02, 0.03,0.03],
                      std1=[0.12,0.21, 0.22,0.14],
                      )
                  )


print(df)


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Number of MADE blocks')
ax1.set_ylabel(r'$FID_M$', color=color)
ax1.errorbar(df['n_made'], df['fid0'], yerr=df['std0'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel(r'$FID_S$', color=color)  # we already handled the x-label with ax1
ax2.errorbar(df['n_made'], df['fid1'], yerr=df['std1'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.title('Influence of the number of MADE blocks')
plt.savefig('n_made_jnfd_ms_fid.pdf')

