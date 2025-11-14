import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from glob import glob

#sns.set_theme(rc={'figure.figsize':(15,7.5)})
#sns.set_style("whitegrid")
sns.set_context("paper", font_scale=0.65)

# load pickle files into dataset
pkl_dir = '.'
lst = glob(os.path.join(pkl_dir,'SN*'))
df = pd.read_pickle(lst[0])
if len(lst)>1:
    for f in lst:
        df = pd.concat([df,pd.read_pickle(f)],ignore_index=True)

# date picker
if ('sd' not in st.session_state) or ('ed' not in st.session_state):
    st.session_state.ed = datetime.now()+timedelta(days=1)
    st.session_state.sd = datetime.now()-timedelta(days=31)

sd = st.date_input(label="MPC From", value=datetime.now()-timedelta(days=31), max_value=st.session_state.ed)
ed = st.date_input(label="MPC To", value=datetime.now(), min_value=sd+timedelta(days=1))

# metric selection
machines = ['SN1616','SN4055']
energies = ['6X','10X','6XFFF','10XFFF','6E','9E','12E','16E']
metrics = ['BeamOutputChange',
           'BeamUniformityChange',
           'BeamCenterShift',
           'IsoCenterSize',
           'IsoCenterMVOffset',
           'IsoCenterKVOffset',
           'MLCMaxOffsetA',
           'MLCMaxOffsetB',
           'MLCMeanOffsetA',
           'MLCMeanOffsetB',
           'MLCBacklashMaxA',
           'MLCBacklashMaxB',
           'MLCBacklashMeanA',
           'MLCBacklashMeanB',
           'JawX1',
           'JawX2',
           'JawY1',
           'JawY2',
           'JawParallelismX1',
           'JawParallelismX2',
           'JawParallelismY1',
           'JawParallelismY2',
           'CollimationRotationOffset',
           'GantryAbsolute',
           'GantryRelative',
           'CouchLat',
           'CouchLng',
           'CouchVrtLarge',
           'CouchRtn',
           'RotationInducedCouchShift']

opt_energy = st.multiselect(
    "Select Energies",
    energies,
    default='6X',
)
opt_metric = st.selectbox(
    "Select Metric",
    metrics,
)

#plt_df = df[['machineName','Energy', 'TimeStamp']+[opt_metric]].between_time(sd,ed)
plt_df = df[['machineName','TimeStamp','Energy']+[opt_metric]]
plt_df = plt_df[(plt_df.Energy.isin(opt_energy)) & (plt_df.TimeStamp >= str(sd)) & (plt_df.TimeStamp <= str(ed))]

fig,axs = plt.subplots(nrows=2)
# TB1 plot
sns.scatterplot(data=plt_df[plt_df.machineName=='SN1616'].drop('machineName',axis=1),x='TimeStamp', y=opt_metric, hue='Energy',ax=axs[0])
# TB2 plot
sns.scatterplot(data=plt_df[plt_df.machineName=='SN4055'].drop('machineName',axis=1),x='TimeStamp', y=opt_metric, hue='Energy',ax=axs[1])
st.pyplot(fig)
