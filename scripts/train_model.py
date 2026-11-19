from pathlib import Path
import json, pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from xgboost import XGBClassifier
r=Path(__file__).resolve().parents[1];a=r/'artifacts';m=r/'models';m.mkdir(exist_ok=True)
X=pd.read_csv(a/'X_train.csv');Xt=pd.read_csv(a/'X_test.csv');y=pd.read_csv(a/'y_train.csv').iloc[:,0];yt=pd.read_csv(a/'y_test.csv').iloc[:,0]
models={
 'Decision Tree':(DecisionTreeClassifier(random_state=42),{'model__max_depth':[4,8],'model__min_samples_leaf':[5,20]}),
 'Bagging':(BaggingClassifier(random_state=42,n_jobs=1),{'model__n_estimators':[15,25],'model__max_samples':[0.8,1.0]}),
 'Random Forest':(RandomForestClassifier(random_state=42,n_jobs=1),{'model__n_estimators':[20,35],'model__max_depth':[8,12]}),
 'AdaBoost':(AdaBoostClassifier(random_state=42),{'model__n_estimators':[20,35],'model__learning_rate':[0.05,0.2]}),
 'Gradient Boosting':(GradientBoostingClassifier(random_state=42),{'model__n_estimators':[20,35],'model__learning_rate':[0.05,0.1]}),
 'XGBoost':(XGBClassifier(random_state=42,n_jobs=1,eval_metric='logloss',tree_method='hist'),{'model__n_estimators':[20,35],'model__max_depth':[3,5]})}
rows=[];best=None;bestname=None;bestscore=-1
for name,(est,grid) in models.items():
 s=GridSearchCV(Pipeline([('imputer',SimpleImputer(strategy='median')),('model',est)]),grid,cv=StratifiedKFold(3,shuffle=True,random_state=42),scoring='f1',n_jobs=1);s.fit(X,y)
 rows.append({'model':name,'cv_f1_mean':s.best_score_,'best_parameters':json.dumps(s.best_params_)})
 if s.best_score_>bestscore:best=s.best_estimator_;bestname=name;bestscore=s.best_score_;params=s.best_params_
pd.DataFrame(rows).to_csv(a/'workflow_experiments.csv',index=False)
pred=best.predict(Xt);result={'selected_model':bestname,'best_parameters':params,'cv_f1':bestscore,'accuracy':accuracy_score(yt,pred),'precision':precision_score(yt,pred),'recall':recall_score(yt,pred),'f1':f1_score(yt,pred),'confusion_matrix':confusion_matrix(yt,pred).tolist()}
dump(best,m/'best_model.joblib');(a/'pipeline_evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
