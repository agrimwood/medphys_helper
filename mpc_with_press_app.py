import os
import csv
import urllib.request
import json
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
from glob import glob

### Function to load MPC directories from config file
def load_config(file_path=''):    
    # Read the JSON file
    with open(file_path, 'r') as file:
        config = json.load(file)
    # Access Machines and MPC_archive
    machines = config['Machines']
    mpc_archive = config['MPC_archive']
    return machines, mpc_archive
    

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


### Collect atompspheric pressure data from opensource api
def load_press():
    '''
        Retrieve atmospheric pressure time series via API and convert to dataframe
    '''
    ApiQuery='https://api.open-meteo.com/v1/forecast?latitude=51.5053&longitude=0.0553&hourly=surface_pressure&past_days=7'
    WEBData = urllib.request.urlopen(ApiQuery)
    apidata = WEBData.read()
    encoding = WEBData.info().get_content_charset('utf-8')
    data = json.loads(apidata.decode(encoding))
    # Convert to Pandas
    press_data = {'TimeStamp': [], 'Values': []}
    for t,v in zip(data['hourly']['time'],data['hourly']['surface_pressure']):
        press_data['TimeStamp'].append(datetime.strptime(t,'%Y-%m-%dT%H:%M'))
        press_data['Values'].append(v)
    press_data = pd.DataFrame.from_dict(press_data)
    return press_data


### Data Visualisation Functions
def tb_plot(machine='',tb_df=pd.DataFrame(),start_dt=None,end_dt=None):
    '''
        Plot selected MPC data for specified linac
    '''
    base = alt.Chart(tb_df[tb_df.machineName==machine].drop('machineName',axis=1),title=alt.Title(machine,anchor='middle'))
    tb_chart = base.mark_line(point=True).encode(
        alt.X('TimeStamp:T', title='Date').scale(domain=(str(start_dt),str(end_dt))),
        y=tb_df.columns.to_list()[-1],
        color=alt.Color('Energy:N').legend(orient='bottom')
    ).properties().interactive(bind_x=False)
    tb_line = base.mark_rule(color='black',size=0.5,strokeDash=[6,3]).encode(
        x=alt.datum(alt.DateTime(date=datetime.now().day,month=datetime.now().month,year=datetime.now().year,hours=datetime.now().hour)))
    return tb_chart+tb_line

def press_plot(press_df=pd.DataFrame(),start_dt=None,end_dt=None):
    '''
        Plot atmospheric pressure data
    '''
    pbase = alt.Chart(press_df,title=alt.Title('Atomspheric Pressure (open-meteo.com)',anchor='middle'))
    p_chart = pbase.mark_line(point=False).encode(
        alt.X('TimeStamp:T', title='Date').scale(domain=(str(start_dt),str(end_dt))),
        alt.Y('Values:Q', title='Surface Pressure (hPa)').scale(domain=(press_df.Values.min(),press_df.Values.max()))
        ).properties().interactive(bind_x=False)
    pline = pbase.mark_rule(color='black',size=0.5,strokeDash=[6,3]).encode(
        x=alt.datum(alt.DateTime(date=datetime.now().day,month=datetime.now().month,year=datetime.now().year,hours=datetime.now().hour)))
    return p_chart+pline


### Function helps to preserve MPC data in session_state
def form_callback(val):
    '''
        Re-initialise session_state if 'Y'
    '''
    if val[0]=='Y':
        st.session_state['initialized'] = True
    else:
        del st.session_state['initialized']


### Main Function
def main():
        
    # mpc data filepaths
    mdirs,pkl_dir = load_config("V:\Medical Physics\Physics\Guys Physics\MPC Offline Analysis\MPC_data\MPC_app_config.json")
    
    # page title
    st.header("MPC Checks")

    # scan for MPC data when starting or refreshing streamlit app
    if 'initialized' not in st.session_state:
        with st.status("Loading MPC records...", expanded=False) as mpcStatus:
            for k in mdirs.keys():
                st.write("Scanning "+k+" records...")
                mpc_dir = mdirs[k]
                #pkl_dir = 'pickles'
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
        
        # load pressure data
        pdata = load_press()
        st.session_state.pdata = pdata
    else:
        # preserve MPC dataframe and pressure data
        with st.status("Updating MPC plots...", expanded=False) as mpcStatus:
            mpcStatus.update(label="MPC plots updated...", state="complete", expanded=False)
        df = st.session_state.df  
        pdata = st.session_state.pdata

    # refresh MPC data
    st.button('Refresh MPC Data',on_click=form_callback,args=['N'])
    
    # date picker
    if ('sd' not in st.session_state) or ('ed' not in st.session_state):
        st.session_state.ed = datetime.now()+timedelta(days=1)
        st.session_state.sd = datetime.now()-timedelta(days=31)
    sd = st.date_input(label="MPC From",
                       value=datetime.now()-timedelta(days=31),
                       max_value=st.session_state.ed-timedelta(days=1),
                       on_change=form_callback,
                       args=['Y'],
                       key='Date From')
    ed = st.date_input(label="MPC To",
                       value=datetime.now()+timedelta(days=1),
                       min_value=sd+timedelta(days=1),
                       on_change=form_callback,
                       args=['Y'],
                       key='Date To')

    # Select results for display
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
        key='Energy',
        on_change=form_callback,
        args=['Y'],
    )
    opt_metric = st.selectbox(
        "Select Metric",
        metrics,
        key='Metric',
    )
    plt_df = df[['machineName','TimeStamp','Energy']+[opt_metric]]
    plt_df = plt_df[(plt_df.Energy.isin(opt_energy)) & (plt_df.TimeStamp >= str(sd)) & (plt_df.TimeStamp <= str(ed))]

    # Data Visulisation
    st.altair_chart(tb_plot('SN1616',plt_df,sd,ed))
    st.altair_chart(tb_plot('SN4055',plt_df,sd,ed))
    st.altair_chart(press_plot(pdata,sd,ed))


if __name__ == "__main__":
    main()
