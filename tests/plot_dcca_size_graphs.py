import seaborn as sb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame( 
                  dict(
                      size = [2,9,16], 
                      acc01 = [0.7518,0.8129,0.7977], 
                      acc10 = [0.5403,0.7875,0.7839], 
                      std10= [0.0007,0.001,0.0008],
                      std01=[0.001,0.001,0.002],
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
