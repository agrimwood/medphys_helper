from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats
import os
import pandas as pd
import numpy as np
import seaborn as sns
from glob import glob

class margin_calc:
  '''
    Class reads exported Offline Review data from Varian ARIA and calculates PTV margins from setup errors.
  '''
  def __init__(self,pt_csv=None,data_dir=None, colab=True):
    '''
      Input
        - pt_csv:        csv path, file must contain the following columns: PatientID, Primary Oncologist, Clinic
        - data_dir:      direcotry path containing ARIA-exported files with first 12 lines removed for anonymisation purposes. File names must match PatientIDs
        - colab (bool):  if True, data is automatically located and saved in Google colab directory structure

      Class variables
        - data_table:    pandas dataframe containing imported offline review data
        - ID_table:      pandas dataframe imported from pt_csv
    '''
    # read pt ID file
    if colab:
      ID_table = pd.read_csv('/content/AnonTable.csv', header=0)
    else:
      ID_table = pd.read_csv(pt_csv, header=0)

    # read pt data files
    if colab:
      file_list = glob('/content/*.txt')
    else:
      file_list = glob(data_dir+os.sep+'*.txt')
    df = pd.DataFrame()
    for f in file_list:
      file_df = pd.read_csv(f, sep='\t',header=0)
      file_df['PatientID'] = int(Path(f).name[:-4])
      df = pd.concat([df, file_df],ignore_index=True)

    # convert couch lateral values to coordinate frame centred at 0cm Lat
    df['Treatment Pos Lat'] = df['Treatment Pos Lat'] - np.where(df['Treatment Pos Lat'] > 50, 1000, 0)
    df['Couch Lat'] = df['Couch Lat'] - np.where(df['Couch Lat'] > 50, 1000, 0)

    # join patient ID table and data tables
    df = df.join(ID_table.set_index('PatientID'), on='PatientID')
    df = df.sort_values(by=['PatientID','Session Date','Session Time'],ignore_index=True)

    # assign class vars
    self.data_table = df
    self.ID_table = ID_table
    self.systematic_err = {}
    self.random_err = {}
    self.colab = colab


  ## Class methods
  # filter and optionally recalc match
  def preprocess_data(self,
                 keep_dict={},
                 drop_dict={},
                 str_dict={},
                 keep_strdict={},
                 split_field=None,
                 split_values=None,
                 fraction_threshold=10,
                 end_to_end=True):
    '''
      Filer and cleanse data loaded from anonymised Offline Review txt files.
      
      Input
        - keep_dict:          (dict) lists by column name. Only rows with listed values are kept
        - drop_dict:          (dict) lists by column name. Rows containing listed values are dropped
        - str_dict:           (dict) lists by column name. Rows containing listed substrings are dropped
        - keep_strdict:       (dict) lists by column name. Only containing listed substrings are kept
        - fraction_threshold: (int) PatientIDs with fewer records are dropped (default 10)
        - end_to_end:         (default True) dictates whether first (True), last (False) or all (None) records per PatientID are used to calculate setup errors

      Class vars
        - filtered_data:      (dataframe) table of filtered data
    '''
    df = self.data_table
    cleaned_df = self.filter_df(df,keep_dict,drop_dict,str_dict,keep_strdict,split_field,split_values,fraction_threshold)
    # recalculate online shifts as difference between treatment position vs couch position
    recalced_df = self.recalc_shifts(cleaned_df, end_to_end)

    # assign class vars
    self.filtered_data = recalced_df
    if keep_dict:
      self.keep_dict = keep_dict
    if drop_dict:
      self.drop_dict = drop_dict
    if str_dict:
      self.str_dict = str_dict
    if split_field:
      self.split_field = split_field
    if split_values:
      self.split_values = split_values
    self.end_to_end = end_to_end


  # calculate systematic and random errors
  def calc_errors(self, error_type='setup'):
    '''
      Calculate setup errors along each axis

      Class vars
        - systematic_err:  (dataframe) systematic setup error in cm
        - random_err:      (dataframe) random error in cm
        - metrics:         (dict) setup errors on individual and population levels
    '''
    df = self.filtered_data
    setup_metrics  = self.setup_errors(df)
    systematic_err = setup_metrics['systematic_pop']
    random_err = setup_metrics['random_pop']

    # assign class vars
    self.systematic_err[error_type] = systematic_err
    self.random_err[error_type] = random_err
    self.metrics = {error_type: setup_metrics}


  # calculte PTV margins from errors
  def calc_margins(self, alpha=2.5, beta=0.7, penumbra=None):
    '''
      Calculate Van Herk PTV margins along 3 axes from setup errors

      Input
        - alpha:    systematic error coefficient (default 2.5)
        - beta:     random error coefficient (default 0.7)
        - penumbra: optional beam penumbral width in cm (default None)
    '''
    systematic_list = []
    random_list = []
    systematic_err = self.systematic_err
    random_err = self.random_err
    for vals in systematic_err.values():
      systematic_list.append(vals)
    for vals in random_err.values():
      random_list.append(vals)

    df_systematic = pd.DataFrame(systematic_list)
    df_random = pd.DataFrame(random_list)
    M = self.van_herk(systematic_errors=df_systematic,random_errors=df_random,alpha=alpha, beta=beta,sigma_p=penumbra)
    self.Margins = M
  

  def plot_setup_errs(self,file_prefix='',match_range=[-1.5,1.5],bin_width=0.05):
    '''
      Plots setup errors as scatter plots in 2D and histograms for each axis
    '''
    sr = match_range
    bw = bin_width
    br = [match_range[0]-bw/2,match_range[1]+bw/2]
    df = self.filtered_data
    pop_avgs = self.metrics['setup']['M_pop']
    pop_std = self.metrics['setup']['systematic_pop']
    M = self.Margins

    for c in ['Lng','Vrt']:
      self.setupscat(df=df,
                x_data=df['Online Match Lat'],
                y_data=df['Online Match '+c],
                plot_title='Online Matches Lat-'+c,
                x_label='Lat (cm)',
                y_label=c+' (cm)',
                E=[pop_avgs['Lat'],pop_avgs[c],M['Lat']*2,M[c]*2],
                grouping='PatientID',
                axis_range=sr,
                legend_display=False,
                fname_prefix=file_prefix)
    
    for c in ['Vrt','Lat','Lng']:
      self.setup_hist(hist_data=df['Online Match '+c],
                      hist_mean=pop_avgs[c],
                      hist_std=pop_std[c],
                      bw=bw,br=br,
                      title='Online Matches '+c,
                      x_label=c+' Match (cm)',
                      fname_prefix=file_prefix)


  ## Utility methods
  # filter function
  @staticmethod
  def filter_df(data=None,
                keep_dict={},
                drop_dict={},
                str_dict={},
                keep_strdict={},
                split_field=None,
                split_values=None,
                fraction_threshold=10):
    '''
      Filter ARIA Offline Review raw data dataframe

      Inputs
        - data (dataframe):         ARIA offline review data
        - keep_dict:                keys are data column names, values are lists of values to keep
        - drop_dict:                keys are data column names, values are lists of values to drop
        - str_dict:                 keys are data column names, values are lists of substrings to drop
        - keep_strdict:             keys are data column names, values are lists of substrings to keep
        - split_field (str):        column name
        - split_values (list):      list of values to split dataset by
        - fraction_threshold (int): ignore cases where number of filtered fractions < threshold
    '''
    filtered_data = data

    # drop rows containing specified substring
    for key, val in str_dict.items():
      if isinstance(val,list):
        regstr = '|'.join(val)
      else:
        regstr = val
      filtered_data = filtered_data[~filtered_data[key].str.contains(regstr, case=False)]

    # keep rows containing specified substring
    for key, val in keep_strdict.items():
      if isinstance(val,list):
        regstr = '|'.join(val)
      else:
        regstr = val
      filtered_data = filtered_data[filtered_data[key].str.contains(regstr, case=False)]

    # keep specified data
    for key,val in keep_dict.items():
      if isinstance(val,list):
        filtered_data = filtered_data[filtered_data[key].isin(val)]
      else:
        filtered_data = filtered_data[filtered_data[key]==val]

    # drop specified data
    for key,val in drop_dict.items():
      if isinstance(val,list):
        filtered_data = filtered_data[~filtered_data[key].isin(val)]
      else:
        filtered_data = filtered_data[filtered_data[key]!=val]

    # drop cases with too few fractions
    f = filtered_data[['PatientID','Plan ID','Fraction','Session Date','Session Time']].copy() 
    f = f.sort_values(['PatientID','Plan ID','Fraction','Session Time']).groupby(['PatientID','Plan ID', 'Fraction'], as_index=False).first()
    f['count'] = f.groupby('PatientID')['PatientID'].transform('count')
    f = f[f['count']>=fraction_threshold].groupby('PatientID').first()
    f = f.index.to_list()
    filtered_data = filtered_data[filtered_data['PatientID'].isin(f)]

    # split data for comparison
    if split_field and split_values:
      comparators = {}
      for l in split_values:
            comparators[l] = filtered_data[filtered_data[split_field]==l]
      return comparators
    return filtered_data


  @staticmethod
  def recalc_shifts(filtered_data=None, end_to_end=True):
    # select all, first or last image matches from each session
    if end_to_end is None:
      return filtered_data
    elif end_to_end:
      filtered_data = filtered_data.sort_values(['Session Date', 'Time']).groupby(['PatientID', 'Plan ID','Fraction'], as_index=False).first()
      # re-calculate online matches
      filtered_data[['Online Match Vrt','Online Match Lat','Online Match Lng']] = np.array(filtered_data[['Treatment Pos Vrt','Treatment Pos Lat','Treatment Pos Lng']])-np.array((filtered_data[['Couch Vrt','Couch Lat','Couch Lng']]))
    else:
      filtered_data = filtered_data.sort_values(['Session Date', 'Time']).groupby(['PatientID', 'Plan ID','Fraction'], as_index=False).last()
      # re-calculate online matches
      filtered_data[['Online Match Vrt','Online Match Lat','Online Match Lng']] = np.array(filtered_data[['Treatment Pos Vrt','Treatment Pos Lat','Treatment Pos Lng']])-np.array((filtered_data[['Couch Vrt','Couch Lat','Couch Lng']]))
    return filtered_data


  @staticmethod
  def setup_errors(filtered_data=None):
    '''
      Calculate setup errors on filtered dataframe
    '''
    # calculate errors
    # Mean individual systematic error
    mp = filtered_data.groupby('PatientID',as_index=True)[['Online Match Vrt','Online Match Lat','Online Match Lng']].mean().copy()
    mp.rename(columns={'Online Match Vrt': 'Vrt', 'Online Match Lat': 'Lat', 'Online Match Lng': 'Lng'}, inplace=True)

    # Population mean setup error
    Mpop = mp[['Vrt','Lat','Lng']].mean()

    # Population systematic error
    P = len(mp.index)
    S = mp - Mpop
    S = np.square(S)
    S = S.sum()
    S = S / (P-1)
    Sigma_s = np.sqrt(S)

    # Individual random error
    s = filtered_data[['PatientID','Online Match Vrt','Online Match Lat','Online Match Lng']].copy()
    s.rename(columns={'Online Match Vrt': 'Vrt', 'Online Match Lat': 'Lat', 'Online Match Lng': 'Lng'}, inplace=True)
    s = s.join(mp,rsuffix='_mp',on='PatientID')
    s[['Vrt','Lat','Lng']] = np.subtract(np.array(s[['Vrt','Lat','Lng']]),np.array(s[['Vrt_mp','Lat_mp','Lng_mp']]))
    s[['Vrt','Lat','Lng']] = np.square(s[['Vrt','Lat','Lng']])
    s = s.groupby('PatientID',as_index=False).sum()
    s = s.join(filtered_data['PatientID'].value_counts().to_frame(),rsuffix='_np',on='PatientID')
    s.rename(columns={'PatientID_np': 'np'}, inplace=True)
    s[['Vrt','Lat','Lng']] = s[['Vrt','Lat','Lng']].divide(np.array(s['np'])-1, axis='index')
    s[['Vrt','Lat','Lng']] = np.sqrt(s[['Vrt','Lat','Lng']])
    s.set_index('PatientID',inplace=True)
    sigma_p = s[['Vrt','Lat','Lng']]

    # Population random error
    sigma_s = sigma_p.mean()

    # Capture population-level results
    results = {'M_pop': Mpop, 'systematic_ind': mp, 'systematic_pop': Sigma_s, 'random_ind': sigma_p, 'random_pop': sigma_s, 'P': P,'N': len(filtered_data.index)}
    return results


  @staticmethod
  def van_herk(systematic_errors=[],random_errors=[],alpha=2.5, beta=0.7,sigma_p=None):
    '''
      calculate van her margins from setup errors according to BIR guidance
      (Tudor GSJ. et al. 2020 Geometric Uncertainties in Daily Online IGRT: Refining the CTV-PTV Margin for Contemporary Photon Radiotherapy. British Institute of Radiology.)
    '''
    S_total = np.sqrt(np.sum(np.square(systematic_errors)))
    if sigma_p is None:
      s_total = np.sqrt(np.sum(np.square(random_errors)))
      M = alpha*S_total + beta*s_total
    else:
      penumbra = [sigma_p]*3
      penumbra = pd.DataFrame(columns=random_errors.columns, data=[penumbra])
      random_errors = pd.concat([random_errors,penumbra])
      s_total = np.sqrt(np.sum(np.square(random_errors)))
      M = alpha*S_total + beta*(s_total - penumbra)
    return M
  

  @staticmethod
  def setupscat(df=None,x_data='',y_data='',plot_title='',grouping='', axis_range=[-1.5,1.5],legend_display=False, im_format='svg', root_dir=None, save_fig=True, fname_prefix='',x_label='',y_label='', E=None):
    sr = axis_range
    if root_dir:
      fname = os.path.join(root_dir,fname_prefix+'_Scat_'+plot_title+'.'+im_format)
    else:
      fname = '/content/img/'+fname_prefix+'_Scat_'+plot_title+'.'+im_format
    sns.scatterplot(data=df,
                    x=x_data,
                    y=y_data,
                    style=grouping,
                    hue=grouping,
                    legend=legend_display)
    plt.plot(sr,[0,0],'k-',linewidth=0.5)
    plt.plot([0,0],sr,'k-',linewidth=0.5)
    if E:
      ellipse=Ellipse(E[:2], E[2], E[3],linewidth=1,fill=False)
      ax = plt.gca()
      ax.add_patch(ellipse)
      plt.plot(E[0],E[1],'kx')
    plt.xlim(sr)
    plt.ylim(sr)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    if save_fig:
      plt.savefig(fname)
    plt.show()


  @staticmethod
  def setup_hist(hist_data=None,hist_mean=None,hist_std=None,bw=0.05,br=[-1.525,1.525],im_format='svg',root_dir=None, save_fig=True, title=None, x_label=None, fname_prefix=''):
    
    def normal(mean, std, color="black", histmax=None, lw=1.5):
      x = np.linspace(mean-4*std, mean+4*std, 200)
      p = stats.norm.pdf(x, mean, std)
      if histmax:
          p = p*histmax/max(p)
      z = plt.plot(x, p, color, linewidth=lw)

    ax = sns.histplot(data=hist_data, stat='count', binwidth=bw, binrange=br)
    normal(hist_mean, hist_std, histmax=ax.get_ylim()[1])
    if title:
      plt.title(title)
    else:
      plt.title(hist_data.name)
    if x_label:
      plt.xlabel(x_label)
    if root_dir:
      fname = os.path.join(root_dir,fname_prefix+'_DHist_'+title+'.'+im_format)
    else:
      fname = '/content/img/'+fname_prefix+'_DHist_'+title+'.'+im_format
    if save_fig:
      print('SAVED: '+fname)
      plt.savefig(fname)
    plt.show()


  # Display some relevant stats
  def sample_stats(self, to_dict=False):
    a = self.filtered_data.copy()
    b = self.ID_table.copy()
    b.set_index('PatientID', inplace=True)
    a=a.join(b,rsuffix='r',on='PatientID')
    D = len(a['Primary Oncologist'].unique())
    P = len(a['PatientID'].unique())
    N = len(a.index)
    H = a['Clinic'].unique()
    print('# Number of Patients: '+str(P))
    print('# Number of Oncologists: '+str(D))
    print('# Number of Fractions: '+str(N))
    print('# Hospital(s): '+str(H))
    if to_dict:
      return {'Patients': P, 'Oncologists': D, 'Fractions': N, 'Hospitals': H}


  def plot_imagetypes(self,group=None,saveplot=None, colour=sns.palettes.mpl_palette('Dark2')):
    '''
      plot bar chart that counts instances of values in dataframe column

      Inputs
        - group (str):    name of dataframe column
        - saveplot (str): optional filename
    '''
    if type(self.filtered_data) is list:
      for key,val in dict:
        val.groupby(group).size().plot(kind='barh', color=colour)
        plt.gca().spines[['top', 'right',]].set_visible(False)
        if saveplot:
          head,tail = os.path.split(saveplot)
          plt.savefig(head+os.sep+val+'_'+tail)
    else:
      self.filtered_data.groupby(group).size().plot(kind='barh', color=colour)
      plt.gca().spines[['top', 'right',]].set_visible(False)
      if saveplot:
        head,tail = os.path.split(saveplot)
        plt.savefig(saveplot)
