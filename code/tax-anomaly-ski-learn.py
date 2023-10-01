%qtconsole
from mlutils import dataset
df3 = dataset.load(name="TaxBQConnector", query="SELECT VISIT_NBR,VISIT_DATE,OTHER_INCOME_IND,WIC_REDEEMED_IND,UPDATED_ITEM_UPC_NBR,TXN_TAXCODE,EXMPT_IND,TAX_PCT,TAX_CATEGORY_ID,TAX_SUBCATG_ID,DOTCOM FROM`wmt-buabookkeeping-prod.Sales_Tax.ScanData` where  DOTCOM <> 1 and OTHER_INCOME_IND <> 1 and WIC_REDEEMED_IND <> 1 and EXMPT_IND <> 1 AND TAX_CATEGORY_ID = 1 AND TAX_SUBCATG_ID in (18,26)")

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
df3

random_state = np.random.RandomState(42)
model=IsolationForest(n_estimators=100,max_samples='auto',contamination=float(0.05),random_state=random_state)

model.fit(df3[['TAX_CATEGORY_ID','TAX_SUBCATG_ID','TXN_TAXCODE','TAX_PCT' ]])

print(model.get_params())

df3['scores'] = model.decision_function(df3[['TAX_CATEGORY_ID','TAX_SUBCATG_ID','TXN_TAXCODE','TAX_PCT' ]])
df3['anomaly_score'] = model.predict(df3[['TAX_CATEGORY_ID','TAX_SUBCATG_ID','TXN_TAXCODE','TAX_PCT' ]])

df3[df3['scores']<=0]


accuracy = list(df3['anomaly_score']).count(-1)
print("Accuracy of the model:", accuracy)


