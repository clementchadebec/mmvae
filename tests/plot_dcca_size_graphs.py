import seaborn as sb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame( 
                  dict(
                      size = [2, 5,9,16], 
                      acc01 = [0.7518,0.80367 ,0.8129,0.7977], 
                      acc10 = [0.5403, 0.7296,0.7875,0.7839], 
                      std10= [0.0007, 0.00121,0.001,0.0008],
                      std01=[0.001, 0.00141,0.001,0.002],
                      )
                  )


print(df)

# sb.lineplot(data=df,y='acc01' , x='size', errorbar='std01')
# sb.lineplot(data=df, y='acc10',x='size', errorbar='std10')
plt.errorbar(df['size'], df['acc01'], yerr=df['std01'], label=r'$P_S$')
plt.errorbar(df['size'], df['acc10'], yerr=df['std10'], label=r'$P_M$')
plt.legend()
plt.title('Coherence of the JNF-DCCA model')
plt.xlabel('Dimension of DCCA embeddings')
plt.ylabel('Coherences')
plt.savefig('dcca_size.pdf')



df = pd.DataFrame( 
                  dict(
                      size = [2, 5,9,16], 
                      fid0 = [10.493,10.56021 ,10.305,10.206], 
                      fid1 = [70.249, 70.42865658027456,66.715,69.372], 
                      std0= [0.04542,0.051976888453579624  ,0.03021,0.04965],
                      std1=[0.203, 0.2317150821357825 ,0.08502,0.205],
                      )
                  )


print(df)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Dimension of DCCA embeddings')
ax1.set_ylabel(r'$FID_M$', color=color)
ax1.errorbar(df['size'], df['fid0'], yerr=df['std0'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel(r'$FID_S$', color=color)  # we already handled the x-label with ax1
ax2.errorbar(df['size'], df['fid1'], yerr=df['std1'], color=color)
ax2.tick_params(axis='y', labelcolor=color)
plt.title('FID values of the JNF-DCCA model')

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.savefig('dcca_fid_size.pdf')
