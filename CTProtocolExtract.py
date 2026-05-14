'''
    Alex Grimwood 2026    
    GUI reads exported CT protocol files from GE CT590 and writes to an Excel file
'''

import tkinter as tk
from tkinter import filedialog, messagebox
import os
import string
import re
import pandas as pd
import xml.etree.ElementTree as ET

# example parameter list
parameters = ['isAutoScanEnabled',
              'examCtdi',
              'examDLP',
              'CTDi',
              'DLP',
              'doseEfficiency',
              'seriesType',
              'seriesNumber',
              'kiloVolts',
              'scanFieldOfViewSize',
              'scanFieldOfViewType',
              'groupType',
              'imageThickness',
              'imageThicknessOfScan',
              'pitch',
              'rotationTime',
              'rotationType',
              'scanSpacing',
              'macroRowWidth',
              'scanTime',
              'scoutPlane',
              'vavReconWindowWidth',
              'vavReconWindowLevel',
              'startLocation',
              'endLocation',
              'tableSpeed',
              'numberOfImages',
              'numImagesPerRotation',
              'numberOfScans',
              'isMarScanOn',
              'isAxialHiResEnabled',
              'cineTimeBetweenImages',
              'cineScanTime',
              'isWideViewReconEnabled',
              'seriesDescriptionRecon',
              'reconNumber',
              'PMRnumImagesPerRotation',
              'PMRimageThickness',
              'automaMaxMilliAmps',
              'automaMinMilliAmps',
              'automaReferenceNoiseIndex',
              'automaNoiseIndex',
              'automAFlag',
              'isSmartScanEnabled',
              'groupDelay',
              'displayFieldOfView',
              'algorithm',
              'matrixSize',
              'vavReconUserAnnoLevel',
              'iterativeMode',
              'iterativeConfig',
              ]

def protocol_extract_df(protocol_file='', parameters=[]):
    '''
        Read .proto file from GT CT_590 and write parameters to a pandas dataframe
        
        Args:
            protocol_file (str): protocol file path
            parameters (list): list of parameters to read
            
        Returns:
            dataframe: protocol names, values and levels
            e_dict (dict): Exam CTDI and DLP
    '''
    
    # initialise vars
    parameters = sorted(set(parameters))
    series_count = 0
    group_count = 0
    recon_count = 0
    flag=-1
    column_names=['Parameter','Value','Series','Group','Recon']
    records = []
    k_list = []
    v_list = []
    recon_rows = []
    e_dict = {}

    # helper function for counting and flagging sections
    def counts(countstr='',count=0):
        count+=1
        countname = countstr+'_'+str(count)
        flag = ['Series','Group','Recon'].index(countstr)+1
        return countname, count, flag

    # read protocol file and parse line by line
    inputfile = open(protocol_file)
    for line in inputfile:
        if line.strip().split(' = ')[0] in parameters:
            [k,v] = line.strip().replace('"','').split(' = ')
            if flag == 0:
                e_dict[k] = v
            elif flag == 3:
                k_list.append(k)
                v_list.append(v)
                recon_rows.append([k,v,sname,gname,rname])
            else:
                records.append([k,v,sname,gname,rname])        
        if line == '\tDoseCheckExamLevelValue {\n': # flag 0 CTDI and DLP
            flag = 0                     
        if line == '\tSeries {\n':
            group_count = 0
            recon_count = 0
            sname, series_count, flag = counts('Series',series_count) # flag 1 Series
            gname = ''
            rname = ''     
        if line == '\t\tGroup {\n':
            recon_count = 0
            gname, group_count, flag = counts('Group',group_count) # flag 2 Group
            rname = ''  
        if line == '\t\t\tRecon {\n':
            rname, recon_count, flag = counts('Recon',recon_count) # flag 3 Recon
            recon_rows = []
            k_list = []
            v_list = []
        if line == '\t\t\t}\n' and flag==3:
            flag=-99
            if v_list[k_list.index('seriesDescriptionRecon')] != '': # discard empty recon entries
                for r in recon_rows:
                    records.append(r)
    inputfile.close()
    
    return pd.DataFrame(records, columns=column_names), e_dict
    

def read_protocols(xml_path):
    '''
        Read .xml protocol directory exported with .proto files from GE CT590
        
        Args:
            xml_path (str): xml file path
        
        Returns:
            records (dict): dictionary of .proto file details e.g. name and file path
    '''
      
    tree = ET.parse(xml_path)
    root = tree.getroot()
    records = {'meta': {'root': os.path.split(xml_path)[0], 'xml_file': xml_path}}
    for category in root.findall("category"):
        category_id = category.get("id")
        category_name = category.get("name")        
        for protocol in category.findall("protocol"):
            protocol_id = protocol.get("id")
            protocol_file = protocol.get("file")
            protocol_name = protocol.get("name")            
            records[protocol_name] = {'path': protocol_file, 'category': category_name, 'index': str(category_id)+'.'+str(protocol_id)}
    
    return records  


def protocols_to_xl(xml_path='', outfile='', parameters=[]):
    '''
        Read GE CT590 protocol xml directory, locate and extract protocols from .proto files, write data to an Excel file
        
        Args:
            xml_path (str): xml read file path
            outfile (str): .xlsx write file path
            parameters (list): protocol parameters
    '''
    
    # create parameters list if absent
    if len(parameters)==0:
        parameters = ['isAutoScanEnabled',
                      'examCtdi',
                      'examDLP',
                      'CTDi',
                      'DLP',
                      'doseEfficiency',
                      'seriesType',
                      'seriesNumber',
                      'kiloVolts',
                      'scanFieldOfViewSize',
                      'scanFieldOfViewType',
                      'groupType',
                      'imageThickness',
                      'imageThicknessOfScan',
                      'pitch',
                      'rotationTime',
                      'rotationType',
                      'scanSpacing',
                      'macroRowWidth',
                      'scanTime',
                      'scoutPlane',
                      'vavReconWindowWidth',
                      'vavReconWindowLevel',
                      'startLocation',
                      'endLocation',
                      'tableSpeed',
                      'numberOfImages',
                      'numImagesPerRotation',
                      'numberOfScans',
                      'isMarScanOn',
                      'isAxialHiResEnabled',
                      'cineTimeBetweenImages',
                      'cineScanTime',
                      'isWideViewReconEnabled',
                      'seriesDescriptionRecon',
                      'reconNumber',
                      'PMRnumImagesPerRotation',
                      'PMRimageThickness',
                      'automaMaxMilliAmps',
                      'automaMinMilliAmps',
                      'automaReferenceNoiseIndex',
                      'automaNoiseIndex',
                      'automAFlag',
                      'isSmartScanEnabled',
                      'groupDelay',
                      'displayFieldOfView',
                      'algorithm',
                      'matrixSize',
                      'vavReconUserAnnoLevel',
                      'iterativeMode',
                      'iterativeConfig',
                      ]
    
    # read xml directory
    records = read_protocols(xml_path)
    protocols = list(records.keys())
    protocols.remove('meta')
    protocol_tables = []
    exam_doses = []
    content_list = []
    print('###################################################')
    
    # loop through protocol files and extract to dataframes
    for protocol in protocols:   
        record = records[protocol]
        fname = os.path.join(records['meta']['root'],record['path'])
        print(r'Processing: '+fname) 
        proto_params, exam_dose = protocol_extract_df(fname,parameters)
        protocol_tables.append(proto_params)
        exam_doses.append(exam_dose)
        content_list.append([record['index'],protocol,record['category'],exam_dose['examCtdi'],exam_dose['examDLP']])

    # write dataframes to worksheets in excel file
    print('\n\nWriting to: '+outfile)
    content_df = pd.DataFrame(content_list,columns=['Index','Protocol','Category','Exam CTDI','Exam DLP'])
    with pd.ExcelWriter(outfile, mode='w') as writer:
        content_df.to_excel(writer, sheet_name='Contents', index=False)
        for i,p in enumerate(protocol_tables):
            print('  '+content_list[i][0]+' '+content_list[i][1])
            p.to_excel(writer, sheet_name=content_list[i][0], index=False)
    print('####### Done... #######')



# ---- GUI Application ----
class ProtocolsApp:
    '''
        Simple GUI wrapper for specifying xml input file path and .xlsx write file path
    '''
    
    def __init__(self, root):
        self.root = root
        self.root.title("CT Protocols to Excel Converter")

        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()

        self._build_ui()

    def _build_ui(self):
        # Input XML
        tk.Label(self.root, text="Input XML File:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        tk.Entry(self.root, textvariable=self.input_file, width=50).grid(row=0, column=1, padx=10, pady=5)
        tk.Button(self.root, text="Browse...", command=self.browse_input).grid(row=0, column=2, padx=10, pady=5)

        # Output XLSX
        tk.Label(self.root, text="Output Excel File:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        tk.Entry(self.root, textvariable=self.output_file, width=50).grid(row=1, column=1, padx=10, pady=5)
        tk.Button(self.root, text="Browse...", command=self.browse_output).grid(row=1, column=2, padx=10, pady=5)

        # Run button
        tk.Button(self.root, text="Run", command=self.run_conversion, bg="#4CAF50", fg="white", width=20)\
            .grid(row=2, column=1, pady=15)

        # Status label
        self.status_label = tk.Label(self.root, text="", fg="blue")
        self.status_label.grid(row=3, column=0, columnspan=3, pady=5)

    def browse_input(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("XML files", "*.xml"), ("All files", "*.*")]
        )
        if file_path:
            self.input_file.set(file_path)

    def browse_output(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if file_path:
            self.output_file.set(file_path)

    def run_conversion(self):
        input_path = self.input_file.get()
        output_path = self.output_file.get()

        # Validation
        if not input_path:
            messagebox.showerror("Error", "Please select an input XML file.")
            return
        if not os.path.exists(input_path):
            messagebox.showerror("Error", "Input file does not exist.")
            return
        if not output_path:
            messagebox.showerror("Error", "Please specify an output XLSX file.")
            return

        try:
            self.status_label.config(text="Processing...", fg="blue")
            self.root.update_idletasks()

            protocols_to_xl(input_path, output_path)

            self.status_label.config(text="Completed successfully.", fg="green")
            messagebox.showinfo("Success", "Conversion completed successfully.")

        except Exception as e:
            self.status_label.config(text="Error occurred.", fg="red")
            messagebox.showerror("Error", str(e))


# ---- Main ----
if __name__ == "__main__":
    root = tk.Tk()
    app = ProtocolsApp(root)
    root.mainloop()