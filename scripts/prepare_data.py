from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
r=Path(__file__).resolve().parents[1]; d=pd.read_csv(r/'data'/'engine_data.csv'); y=d['Engine Condition']; X=d.drop(columns=['Engine Condition'])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
a=r/'artifacts';a.mkdir(exist_ok=True)
for n,v in [('X_train',Xtr),('X_test',Xte),('y_train',ytr.to_frame()),('y_test',yte.to_frame())]:v.to_csv(a/f'{n}.csv',index=False)
print('Train:',len(Xtr),'Test:',len(Xte))
