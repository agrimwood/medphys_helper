import os
import csv
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
from glob import glob


### Function to update MPC data
def mpcupdate(mpc_dir,pkl_dir,machine_name):
    '''
        Collect and add MPC data to pickles if not already present
    '''
    # read dataframe archive and record the latest Timestamp
    lst = glob(os.path.join(pkl_dir,machine_name+'_*'))
    ymax = max([int(y[-4::]) for y in lst])
    tb_dt = pd.read_pickle(os.path.join(pkl_dir,machine_name+"_"+str(ymax))).sort_values(by=["machineName","TimeStamp","Sequence"],ignore_index=True, ascending=False).TimeStamp.max()
    
    # initialise data dict
    metrics = ['IsoCenterSize',
            'IsoCenterMVOffset',
            'IsoCenterKVOffset',
            'BeamOutputChange',
            'BeamUniformityChange',
            'BeamCenterShift',
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

    # initialise results dicts
    data = {'machineName': [], 'TimeStamp': [], 'Sequence': [], 'Type': [], 'Energy': []}
    for m in metrics:
        data[m] = []  
    y0=ymax
    for l in sorted(os.listdir(mpc_dir), reverse=True):
        paths = os.path.join(mpc_dir,l)
        f = os.path.join(paths,'Results.csv')
        p = os.path.normpath(f)
        p = p.split(os.sep)
        p = p[-2].split('-')
        try:
            tStamp = datetime.strptime('-'.join(p[3:6])+' '+':'.join(p[6:9]), '%Y-%m-%d %H:%M:%S')
        except:
            pass
        else:
            if tStamp>tb_dt and p[2]==machine_name and os.path.isfile(f):
                try:                
                    machineName = p[2]
                    seq = int(p[-2])
                    tmplt = p[-1].split('Template')
                    mpc_type = tmplt[0]
                    energy = tmplt[1].split('MV')[0].upper()
                except:
                    print(paths)
                else:
                    y=tStamp.year
                    if y!=y0:
                        print([y,y0])
                        # write remaining data to dataframe and pickle
                        df = pd.DataFrame(data)
                        if len(df)>0:
                            if y0==ymax:
                                df_prev = pd.read_pickle(os.path.join(pkl_dir,machine_name+"_"+str(ymax))).sort_values(by=["machineName","TimeStamp","Sequence"],ignore_index=True, ascending=False)
                                df = pd.concat([df,df_prev], ignore_index=True)
                            df.sort_values(by=["machineName","TimeStamp","Sequence"],ignore_index=True,inplace=True,ascending=False)
                            df.to_pickle(os.path.join(pkl_dir,machineName+'_'+str(df.TimeStamp.dt.year[0])))
                        data = {'machineName': [], 'TimeStamp': [], 'Sequence': [], 'Type': [], 'Energy': []}
                        for m in metrics:
                            data[m] = []
                        y0=y               
                    
                    # extract file metadata
                    p = os.path.normpath(f)
                    p = p.split(os.sep)
                    p = p[-2].split('-')
                    
                    # metadata to dicts
                    data['machineName'].append(machineName)
                    data['TimeStamp'].append(tStamp)
                    data['Sequence'].append(seq)
                    data['Type'].append(mpc_type)
                    data['Energy'].append(energy)
                    
                    # extract test results from files
                    m_flags = {}
                    for m in metrics:
                        m_flags[m] = 0
                    with open(f, 'r', errors='replace') as csvfile:
                        reader = csv.reader(csvfile)
                        for line in reader:
                            test_name = line[0].split('/')[-1].split(' [')[0]
                            if test_name in metrics and m_flags[test_name]==0:
                                m_flags[test_name] = 1
                                data[test_name].append(float(line[1].replace(' ','')))
                        for m in metrics:
                            if m_flags[m] == 0:
                                data[m].append(np.nan)
            elif tStamp <= tb_dt:
                break
    # write remaining data to dataframe and pickle
    df = pd.DataFrame(data)
    if len(df)>0:
        if y0==ymax:
            df_prev = pd.read_pickle(os.path.join(pkl_dir,machine_name+"_"+str(ymax))).sort_values(by=["machineName","TimeStamp","Sequence"],ignore_index=True, ascending=False)
            df = pd.concat([df,df_prev], ignore_index=True)
        df.sort_values(by=["machineName","TimeStamp","Sequence"],ignore_index=True,inplace=True,ascending=False)
        for y in df.TimeStamp.dt.year.unique():
            df[df.TimeStamp.dt.year==y].to_pickle(os.path.join(pkl_dir,machineName+'_'+str(df.TimeStamp.dt.year[0])))

###

def main():
    # mpc data filepaths
    mdirs = {"SN1616": r'\\XXXXXXXXXXXX\VA_transfer\TDS\H191616\MPCChecks',"SN4055": r'\\XXXXXXXXXXXXXXXX\VA_transfer\TDS\H194055\MPCChecks'}
    pkl_dir = os.path.join("V:","Medical Physics","Physics","Guys Physics","MPC Offline Analysis","MPC_data")
    
    # page title
    st.header("MPC Checks")

    # scan for MPC data when starting streamlit app
    if 'initialized' not in st.session_state:
        with st.status("Loading MPC records...", expanded=False) as mpcStatus:
            for k in mdirs.keys():
                st.write("Scanning "+k+" records...")
                mpc_dir = mdirs[k]
                pkl_dir = 'pickles'
                mpcupdate(mpc_dir,pkl_dir,k)

            # load pickle files
            lst = glob(os.path.join(pkl_dir,'SN*'))
            df = pd.read_pickle(lst[0])
            if len(lst)>1:
                for f in lst:
                    st.write('Loading data: "'+f+'"...')
                    df = pd.concat([df,pd.read_pickle(f)],ignore_index=True)
            mpcStatus.update(label="MPC records successfully loaded...", state="complete", expanded=False)
        st.session_state.initialized = True
        st.session_state.df = df
    else:
        # preserve MPC dataframe 
        with st.status("Updating MPC records...", expanded=False) as mpcStatus:
            mpcStatus.update(label="MPC records updated...", state="complete", expanded=False)
        df = st.session_state.df      

    # date picker
    if ('sd' not in st.session_state) or ('ed' not in st.session_state):
        st.session_state.ed = datetime.now()+timedelta(days=1)
        st.session_state.sd = datetime.now()-timedelta(days=31)
    sd = st.date_input(label="MPC From", value=datetime.now()-timedelta(days=31), max_value=st.session_state.ed-timedelta(days=1), key='Date From')
    ed = st.date_input(label="MPC To", value=datetime.now()+timedelta(days=1), min_value=sd+timedelta(days=1), key='Date To')

    # metric selection
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
        key='Energy'
    )
    opt_metric = st.selectbox(
        "Select Metric",
        metrics,
        key='Metric'
    )

    plt_df = df[['machineName','TimeStamp','Energy']+[opt_metric]]
    plt_df = plt_df[(plt_df.Energy.isin(opt_energy)) & (plt_df.TimeStamp >= str(sd)) & (plt_df.TimeStamp <= str(ed))]

    tb1_chart = alt.Chart(plt_df[plt_df.machineName=='SN1616'].drop('machineName',axis=1),title=alt.Title('SN1616',anchor='middle')).mark_line(point=True).encode(
        x='TimeStamp',
        y=opt_metric,
        color='Energy',
    ).interactive(bind_x=False)
    tb2_chart = alt.Chart(plt_df[plt_df.machineName=='SN4055'].drop('machineName',axis=1),title=alt.Title('SN4055',anchor='middle')).mark_line(point=True).encode(
        x='TimeStamp',
        y=opt_metric,
        color='Energy'
    ).properties().interactive(bind_x=False)

    st.altair_chart(tb1_chart)
    st.altair_chart(tb2_chart)


if __name__ == "__main__":
    main()
